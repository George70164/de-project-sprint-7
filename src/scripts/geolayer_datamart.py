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
    # Входные параметры
    date = sys.argv[1]  # "2022-01-02"
    path_to_geo_events = sys.argv[2]  # "/user/master/data/geo/events"
    path_to_geo_city = sys.argv[3]  # "/user/gera190770/data/geo/geo_time_zone.csv"
    output_base_path = sys.argv[4]  # "/user/gera190770/analytics/geolayer_datamart"

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

    # Считаем дистанцию между координатами отправленного сообщения и координатами города
    df = df.withColumn("distance", F.lit(2) * F.lit(6371)
                                            * F.asin(F.sqrt(F.pow(F.sin((F.radians(F.col('lat')) - F.radians(F.col('lat_city')))/F.lit(2)), 2)
                                            + F.cos(F.radians(F.col('lat')))
                                            * F.cos(F.radians(F.col('lat_city')))
                                            * F.pow(F.sin((F.radians(F.col('lon')) - F.radians(F.col('lon_city')))/F.lit(2)), 2))))

    df = df.join(df.select('event', 'distance').groupBy('event').agg(F.min('distance')), 'event')

    df = df.where(F.col('distance') == F.col('min(distance)')).select('event', 'event_type', 'zone_id')

    # Рассчитываем месяц и неделю расчета
    df = df.withColumn('month', F.trunc('event.datetime', 'month'))
    df = df.withColumn('week', F.trunc('event.datetime', 'week'))

    # Фильтруем события по типу подписки
    df_subscription = df.filter(F.col('event_type') == 'subscription')
    df_reaction = df.filter(df['event_type'] == 'reaction')
    df_message = df.filter(df['event_type'] == 'message')

    # Группировка по городу, месяцу и неделе, счетчик событий
    df_subscription_grouped = df_subscription.groupBy('zone_id', 'month', 'week').agg(F.count('*').alias('week_subscription'))
    df_reaction_grouped = df_reaction.groupBy('zone_id', 'month', 'week').agg(F.count('*').alias('week_reaction'))
    df_message_grouped = df_message.groupBy('zone_id', 'month', 'week').agg(F.count('*').alias('week_message'))
    df_user_grouped = df.groupBy('zone_id', 'month', 'week').agg(F.countDistinct('event.message_from').alias('week_user'))

    # Суммирование подписок за месяц
    df_subscription_grouped = df_subscription_grouped.withColumn('month_subscription', F.sum('week_subscription').over(Window.partitionBy('zone_id', 'month')))
    df_reaction_grouped = df_reaction_grouped.withColumn('month_reaction', F.sum('week_reaction').over(Window.partitionBy('zone_id', 'month')))
    df_message_grouped = df_message_grouped.withColumn('month_message', F.sum('week_message').over(Window.partitionBy('zone_id', 'month')))
    df_user_grouped = df_user_grouped.withColumn('month_user', F.sum('week_user').over(Window.partitionBy('zone_id', 'month')))

    # Присоединение результатов витрины
    df_vitrina = df_subscription_grouped.join(df_reaction_grouped, on=['zone_id', 'month', 'week'], how='left')
    df_vitrina = df_vitrina.join(df_message_grouped, on=['zone_id', 'month', 'week'], how='left')
    df_vitrina = df_vitrina.join(df_user_grouped, on=['zone_id', 'month', 'week'], how='left')

    # Вывод результата
    df_vitrina.write.mode("overwrite").parquet(f"{output_base_path}/date={date}")


if __name__ == "__main__":
    main()

# Команда для запуска из терминала !/usr/lib/spark/bin/spark-submit --master yarn --deploy-mode cluster /lessons/geolayer_datamart.py 2022-05-25 /user/master/data/geo/events /user/gera190770/data/geo/geo_time_zone.csv /user/gera190770/analytics/geolayer_datamart