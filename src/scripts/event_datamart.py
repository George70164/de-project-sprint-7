import sys
import os
import pyspark.sql.functions as F

from pyspark.sql import SparkSession
from pyspark.sql.window import Window

os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/bin/python3"
os.environ["HADOOP_CONF_DIR"] = "/etc/hadoop/conf"
os.environ["YARN_CONF_DIR"] = "/etc/hadoop/conf"
os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3"


def change_dec_sep(df, column):
    # Заменяем ',' на '.' в lat, lng
    df = df.withColumn(column, F.regexp_replace(column, ',', '.'))
    df = df.withColumn(column, df[column].cast("float"))
    return df


def main():
    # Входные параметры (для теста)
    date = sys.argv[1]  # "2022-01-02"
    path_to_geo_events = sys.argv[2]  # "/user/master/data/geo/events"
    path_to_geo_city = sys.argv[3]  # "/user/gera190770/data/geo/geo_time_zone.csv"
    output_base_path = sys.argv[4]  # "/user/gera190770/analytics/user_datamart"
    earth_radius = 6371

    # Создаем подключение
    spark = SparkSession.builder.appName("Project-sp7").getOrCreate()

    # Читаем фрейм с координатами городов
    geo_df = spark.read.csv(path_to_geo_city,
                            sep=';',
                            header=True,
                            inferSchema=True)

    # Приводим lat, lng к типу float, предварительно изменяя ',' на '.'
    geo_df = change_dec_sep(geo_df, 'lat')
    geo_df = change_dec_sep(geo_df, 'lng')

    # Читаем фрейм с событиями и их координатами
    events_df = spark.read.parquet(f'{path_to_geo_events}/date={date}')

    # Соединяем два датафрейма 
    df = events_df.crossJoin(geo_df.select(F.col('id').alias('zone_id'),
                                           F.col('city'),
                                           F.col('lat').alias('lat_city'),
                                           F.col('lng').alias('lon_city'),
                                           F.col('timezone')))

    # датафрейм для рассчета домашнего адреса
    events_df_for_home_city = events_df.crossJoin(geo_df.select(F.col('id').alias('zone_id'),
                                                                F.col('city'),
                                                                F.col('lat').alias('lat_city'),
                                                                F.col('lng').alias('lon_city'),
                                                                F.col('timezone')))

    # Считаем дистанцию между координатами отправленного сообщения и координатами города
    df = df.withColumn("distance", F.lit(2) * F.lit(earth_radius)
                                            * F.asin(F.sqrt(F.pow(F.sin((F.radians(F.col('lat')) - F.radians(F.col('lat_city')))/F.lit(2)), 2)
                                            + F.cos(F.radians(F.col('lat')))
                                            * F.cos(F.radians(F.col('lat_city')))
                                            * F.pow(F.sin((F.radians(F.col('lon')) - F.radians(F.col('lon_city')))/F.lit(2)), 2))))

    df = df.select('event', 'event_type', 'zone_id', 'city', 'distance')
    df = df.join(df.select('event', 'distance').groupBy('event').agg(F.min('distance')), 'event')
    df = df.filter((F.col('distance') == F.col('min(distance)')) & (F.col('event.user').isNotNull())) 

    # Определение актуального адреса
    windowSpec = Window.partitionBy("event.user").orderBy(F.desc("event.datetime"))
    df = df.withColumn("row_number", F.row_number().over(windowSpec))
    act_city_df = df.filter(df.row_number == 1).select(F.col('event.user').alias('user_id'), F.col('city').alias('act_city'))
    df = df.drop("row_number")

    # Определение домашнего адреса
    df_for_home_city = events_df_for_home_city.select(F.col('event.message_from').alias('user_id'),"city",F.date_trunc("day",F.coalesce(F.col('event.datetime'),F.col('event.message_ts'))).alias("date"))
    df_home_city = (df_for_home_city.withColumn("row_date",F.row_number().over(
            Window.partitionBy("user_id", "city").orderBy(F.col("date").desc())))
            .filter(F.col("row_date") >= 27)
            .withColumn("row_by_row_date", F.row_number().over(
            Window.partitionBy("user_id").orderBy(F.col("row_date").desc())))
            .filter(F.col("row_by_row_date") == 1)
            .withColumnRenamed("user_id", "user_id")
            .withColumnRenamed("city", "home_city")
            .drop("row_date", "row_by_row_date", "date"))

    # Создание витрины пользователей
    users_vitrina_df = act_city_df.join(df_home_city, "user_id", "left")

    # Вычисление количества посещенных городов и списка городов в порядке посещения
    window_spec2 = Window.partitionBy('event.user').orderBy('event.datetime')
    df_with_window = df.withColumn('lag_city', F.lag('city').over(window_spec2))
    df_with_window = df_with_window.withColumn('lead_city', F.lead('city').over(window_spec2))
    df_border_cities = df_with_window.filter((df_with_window['lag_city'] != df_with_window['city']) | (df_with_window['lead_city'] != df_with_window['city']))
    df_travel_count = df_border_cities.groupBy(F.col('event.user').alias('user_id')).agg(F.count('*').alias('travel_count'))
    df_travel_array = df_border_cities.groupBy(F.col('event.user').alias('user_id')).agg(F.collect_list('city').alias('travel_array'))

    # Присоединяем столбцы df_travel_array и df_travel_count к итоговой витрине
    users_vitrina_df = users_vitrina_df.join(df_travel_count, 'user_id', 'left') \
                                       .join(df_travel_array, 'user_id', 'left')

    # Вычисление местного времени для каждого пользователя
    local_time_df = act_city_df.join(events_df.select(F.col('event.user').alias('user_id'),
                                                      F.date_format('event.datetime', "HH:mm:ss") \
                                                       .alias('time_utc')), 'user_id', "left")
    local_time_df = local_time_df.join(geo_df, local_time_df['act_city'] == geo_df['city'], "left")
    local_time_df = local_time_df.withColumn("local_time", F.from_utc_timestamp(F.col("time_utc"), F.col("timezone")))
    local_time_df = local_time_df.select('user_id', 'local_time')

    # Добавим местное время в итоговую витрину
    users_vitrina_df = users_vitrina_df.join(local_time_df, 'user_id', 'left')

    # Вывод результата
    users_vitrina_df.write.mode("write").parquet(f"{output_base_path}/date={date}")


if __name__ == "__main__":
    main()

# Команда для запуска из терминала !/usr/lib/spark/bin/spark-submit --master local --deploy-mode client /lessons/event_datamart.py 2022-01-02 /user/master/data/geo/events /user/gera190770/data/geo/geo_time_zone.csv /user/gera190770/analytics/user_datamart