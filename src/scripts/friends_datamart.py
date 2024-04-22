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
    output_base_path = sys.argv[4] # "/user/gera190770/analytics/friends_datamart"

    # Создаем подключение
    spark = SparkSession.builder.appName("Project-sp7").getOrCreate()
    spark.conf.set("spark.sql.analyzer.failAmbiguousSelfJoin", "false")
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

    df = df.select('event', 'event_type', 'zone_id', 'city', 'distance', 'lat', 'lon', 'timezone')
    df = df.join(df.select('event', 'distance').groupBy('event').agg(F.min('distance')), 'event')
    df = df.filter((F.col('distance') == F.col('min(distance)')) & (F.col('event.message_from').isNotNull())) # Здесь убираются все строки 

    # Находим список пользователей, подписанных на один и тот же канал
    same_channel_users = df.select('event.subscription_channel', F.col('event.user').alias('user_left'))\
                           .filter(df.event.subscription_channel.isNotNull() & df.event.user.isNotNull())

    same_channel_users = same_channel_users.join(same_channel_users.select('subscription_channel', F.col('user_left').alias('user_right')), 'subscription_channel')
    same_channel_users = same_channel_users.filter(same_channel_users.user_left != same_channel_users.user_right)

    # Находим пользователей, которые уже переписывались друг с другом
    interacted_users = df.groupBy("event.message_from", "event.message_to")\
                         .count().filter(F.col("count") > 1).select(
        F.col("message_from").alias("user_left"),
        F.col("message_to").alias("user_right")
    )

    # Убираем пользователей из списка
    recommended_users = same_channel_users.join(interacted_users,
                                                (same_channel_users["user_left"].contains(interacted_users["user_left"])) &
                                                (same_channel_users["user_right"].contains(interacted_users["user_right"])),
                                                "leftanti")

    # Найдем пользователей, которые находятся на расстоянии 1 км друг от друга
    window_spec = Window.partitionBy('event.user').orderBy(F.col('event.datetime').desc())

    user_geo = df.withColumn('rank', F.row_number().over(window_spec)).filter(F.col('rank') == 1).select('event.user', 'lat', 'lon') # здесь формируется орлдна строка из-за фильтрации по raNK

    joined_df = recommended_users.join(user_geo.select('user', F.col('lat').alias('ul_lat'), F.col('lon').alias('ul_lon')),
                                       recommended_users.user_left == user_geo.user, "left") \
                                               .join(user_geo.select('user', F.col('lat').alias('ur_lat'), F.col('lon').alias('ur_lon')),
                                                     recommended_users.user_right == user_geo.user, "left")

    joined_df = joined_df.withColumn("distance", F.lit(2) * F.lit(6371) 
                                                          * F.asin(F.sqrt(F.pow(F.sin((F.radians(F.col('ul_lat')) - F.radians(F.col('ur_lat')))/F.lit(2)), 2)
                                                          + F.cos(F.radians(F.col('ul_lat'))) * F.cos(F.radians(F.col('ur_lat')))
                                                          * F.pow(F.sin((F.radians(F.col('ul_lon')) - F.radians(F.col('ur_lon')))/F.lit(2)), 2))))

    recommended_users = joined_df.select('user_left', 'user_right').filter(F.col('distance') <= 1)

    # Добавление атрибутов витрины
    recommended_users = recommended_users.withColumn("processed_dttm", F.current_timestamp())

    user_zone = df.select('event.user',  'zone_id') 
    recommended_users = recommended_users.join(user_zone, F.col("user_left") == F.col("user"), "left").drop('user')

    local_time_df = df.select('zone_id', 'timezone', F.date_format('event.datetime', "HH:mm:ss").alias('time_utc'))
    recommended_users = recommended_users.join(local_time_df, recommended_users.zone_id == local_time_df.zone_id)
    recommended_users = recommended_users.withColumn("local_time", F.from_utc_timestamp(F.col("time_utc"), F.col("timezone")))\
                                         .drop('local_time_df.zone_id', 'timezone', 'time_utc')

    # Вывод результата
    recommended_users.write.mode("overwrite").parquet(f"{output_base_path}/date={date}")


if __name__ == "__main__":
    main()


# # !/usr/lib/spark/bin/spark-submit --master yarn --deploy-mode cluster /lessons/friends_datamart.py 2022-01-02 /user/master/data/geo/events /user/gera190770/data/geo/geo_time_zone.csv /user/gera190770/analytics/friends_datamart
