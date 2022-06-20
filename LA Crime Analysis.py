# Databricks notebook source
# DBTITLE 1,Load in plotting packages
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# COMMAND ----------

# DBTITLE 1,Load in original dataset
file_location = "/FileStore/tables/Crime_Data_from_2020_to_Present.csv"
file_type = "csv"

df =spark.read.csv(file_location,header=True,inferSchema=True)
df.show()

# COMMAND ----------

# DBTITLE 1,Load in time occurrence in String
file_location = "/FileStore/tables/Crime_Data_from_2020_to_Present.csv"
file_type = "csv"

df_hour =spark.read.csv(file_location,header=True,inferSchema=False)
df_hour=df_hour.select('TIME OCC')
display(df_hour)

# COMMAND ----------

# DBTITLE 1,Convert time occurrence to hour
from pyspark.sql.functions import hour, to_timestamp

df_hour=df_hour.withColumn('IncidentTime', to_timestamp(df_hour['Time OCC'],'HHmm'))
df_hour=df_hour.withColumn('Hour',hour(df_hour['IncidentTime']))
df_hour=df_hour.select('Hour')
display(df_hour)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# DBTITLE 1,Select the columns we need
df_crime=df.select(["DATE OCC","AREA NAME","Crm Cd Desc","Vict Age","Vict Sex","Vict Descent","Premis Desc","Weapon Desc","Status Desc","LAT","LON"])
display(df_crime)

# COMMAND ----------

# DBTITLE 1,Rename the columns
df_crime=df_crime.withColumnRenamed('DATE OCC', 'date_occ')
df_crime=df_crime.withColumnRenamed('AREA NAME', 'area_name')
df_crime=df_crime.withColumnRenamed('CRM Cd Desc', 'crime_desc')
df_crime=df_crime.withColumnRenamed('VICT Age', 'vict_age')
df_crime=df_crime.withColumnRenamed('VICT Sex', 'vict_sex')
df_crime=df_crime.withColumnRenamed('VICT Descent', 'vict_desc')
df_crime=df_crime.withColumnRenamed('Premis Desc', 'premis_desc')
df_crime=df_crime.withColumnRenamed('Weapon Desc', 'weapon_desc')
df_crime=df_crime.withColumnRenamed('Status Desc', 'status_desc')
df_crime=df_crime.withColumnRenamed('LAT', 'lat')
df_crime=df_crime.withColumnRenamed('LON', 'lon')

# COMMAND ----------

# DBTITLE 1,Extract time-related features
from pyspark.sql.functions import to_date, month, year, dayofweek

spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")
df_crime=df_crime.withColumn("IncidentDate",  to_date(df_crime.date_occ, 'MM/dd/yyy'))
df_crime=df_crime.withColumn('DayofWeek',dayofweek(df_crime['IncidentDate']))
df_crime=df_crime.withColumn('Month',month(df_crime['IncidentDate']))
df_crime=df_crime.withColumn('Year', year(df_crime['IncidentDate']))
display(df_crime)

# COMMAND ----------

# DBTITLE 1,Concatenate 'Hour' column to the whole dataset
from pyspark.sql.functions import monotonically_increasing_id

df_hour = df_hour.withColumn("row_id", monotonically_increasing_id())
df_crime = df_crime.withColumn("row_id", monotonically_increasing_id())
df_crime = df_crime.join(df_hour, ("row_id")).drop("row_id")

df_crime.display()

# COMMAND ----------

df_crime=df_crime.drop('date_occ')
display(df_crime)

# COMMAND ----------

df_crime.createOrReplaceTempView("la_crime")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Part 1: 
# MAGIC Count the number of crimes for different description.

# COMMAND ----------

# DBTITLE 1,Spark dataframe based solution for Part 1
df_1 = df_crime.groupBy('crime_desc').count().orderBy('count', ascending = False)
display(df_1)

# COMMAND ----------

# DBTITLE 1,Spark SQL based solution for Part 1
sql_1 = spark.sql("SELECT crime_desc, COUNT(*) AS Count FROM la_crime GROUP BY crime_desc ORDER BY Count DESC")
display(sql_1)

# COMMAND ----------

# DBTITLE 1,Visualize the result
fig_dims = (20,10)
fig = plt.subplots(figsize = fig_dims)
spark_df_1_plot = df_1.toPandas()
spark_df_1_plot=spark_df_1_plot[spark_df_1_plot['count']>1000]
chart = sns.barplot(x = 'crime_desc', y = 'count', palette= 'BuGn_r', data = spark_df_1_plot )
chart.set_xticklabels(chart.get_xticklabels(), rotation = 90, horizontalalignment = 'right')
display(chart)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Part1_Insight: According to the number of crimes, we can classify crime descriptions into 3 groups based on the above-mentioned graphs and tables: high crime rate, medium crime rate, and low crime rate.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Part 2:
# MAGIC Counts the number of crimes for different area

# COMMAND ----------

# DBTITLE 1,Spark dataframe based solution for Part 2
df_2 = df_crime.groupBy('area_name').count().orderBy('Count', ascending = False)
display(df_2)

# COMMAND ----------

# DBTITLE 1,Spark SQL based solution for Part 2
sql_2 = spark.sql("SELECT area_name, COUNT(*) AS Count FROM la_crime GROUP BY 1 ORDER BY 2 DESC")
display(sql_2)

# COMMAND ----------

# DBTITLE 1,Visualize the result
fig_dims = (20,10)
fig = plt.subplots(figsize = fig_dims)
spark_df_2_plot = df_2.toPandas()
chart = sns.barplot(x = 'area_name', y = 'count', palette= 'BuGn_r',data = spark_df_2_plot )
chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment = 'right')
display(chart)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Part 3:
# MAGIC Count the number of crimes each "Sunday" at "LA downtown".  

# COMMAND ----------

# DBTITLE 1,Spark dataframe based solution for Part 3
la_downtown = (df_crime.lon > -118.2519) & (df_crime.lon < -118.2419) & (df_crime.lat < 34.0500) & (df_crime.lat > 34.0300 )
df_3 = df_crime.filter((df_crime.DayofWeek == 1) & (la_downtown)).groupby('IncidentDate','DayofWeek').count().orderBy('IncidentDate')
display(df_3)

# COMMAND ----------

# DBTITLE 1,Spark SQL based solution for Part 3
sql_3 = spark.sql("SELECT IncidentDate, DayofWeek, COUNT(*) AS Count FROM la_crime WHERE DayofWeek = 1 \
                          AND lon > -118.2519 AND lon < -118.2419 AND lat > 34.0300 AND lat < 34.0500 \
                          GROUP BY IncidentDate, DayofWeek ORDER BY IncidentDate")
display(sql_3)

# COMMAND ----------

# DBTITLE 1,Visualize the result
fig_dims = (20,10)
fig = plt.subplots(figsize = fig_dims)
spark_df_3_plot = df_3.toPandas()
chart = sns.barplot(x = 'IncidentDate', y = 'count',data = spark_df_3_plot )
chart.xaxis.set_major_locator(ticker.MultipleLocator(5))
plt.xticks(rotation=90)
display(chart)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Part 4:
# MAGIC Analysis the number of crime in each month of 2020, 2021, 2022. 

# COMMAND ----------

# DBTITLE 1,Spark dataframe based solution for Part 4
df_4 = df_crime.groupby('Year', 'Month').count().orderBy('Year','Month')
display(df_4)

# COMMAND ----------

# DBTITLE 1,Spark SQL based solution for Part 4
sql_4 = spark.sql('''
                  SELECT Year, 
                         Month, 
                         COUNT(*) AS Count 
                  FROM la_crime 
                  GROUP BY 1, 2 
                  ORDER BY Year, Month
                  ''')
display(sql_4)

# COMMAND ----------

# MAGIC %sql select distinct(crime_desc) as type, count(*) as Count, Year from la_crime group by 1,3 order by 2 desc

# COMMAND ----------

# MAGIC %sql select Year, vict_sex as victim_gender, count(*) as Count from la_crime where (vict_sex='M') or (vict_sex='F') group by 1,2 order by 3 desc

# COMMAND ----------

# MAGIC %sql select Year, vict_age as victim_age, count(*) as Count from la_crime where vict_age != 0 group by 1,2 order by 3 desc

# COMMAND ----------

# MAGIC %sql select Year, vict_desc as victim_descent, count(*) as Count from la_crime where vict_age != 0 group by 1,2 order by 3 desc

# COMMAND ----------

# MAGIC %sql select Year, premis_desc, count(*) as Count from la_crime group by 1,2 order by 3 desc

# COMMAND ----------

# MAGIC %sql select Year, weapon_desc, count(*) as Count from la_crime where weapon_desc is not null group by 1,2 order by 3 desc

# COMMAND ----------

# MAGIC %md
# MAGIC #### Part 4_Business Impact:
# MAGIC 1. The number of crimes dropped from March 2020 when the COVID-19 lockdown started. The number increased again in July 2021.
# MAGIC 2. The most prevalent types of crimes in LA from 2020 to 2022 are stolen vehicle and battery-simple assualt.
# MAGIC 3. The most prevalent victim descent in LA from 2020 to 2022 is Hispanic/Latin/Mexican.
# MAGIC 4. The most prevalent type of weapon in LA from 2020 to 2022 is Strong-ARM (Hands,Fist,Feet or Body Force).

# COMMAND ----------

# MAGIC %md
# MAGIC #### Part 5:
# MAGIC Analysis the number of crime w.r.t the hour in certian day like 2020/01/01, 2021/01/01, 2022/01/01.

# COMMAND ----------

# DBTITLE 1,Spark dataframe based solution for Part 5
dates = ['2020-01-01','2021-01-01','2022-01-01']
df_days = df_crime[df_crime.IncidentDate.isin(dates)]
df_5 = df_days.groupby('Hour','IncidentDate').count().orderBy('IncidentDate','Hour')
display(df_5)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Part 5_Travel Suggestion
# MAGIC It can be clearly seen from the above figure that there are two peak periods of crime, 12 o'clock and 18 o'clock. This time period is the time for tourists to eat and rest. So remind visitors not to relax their vigilance while resting.

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Part 6
# MAGIC - Step1: Find out the top-3 danger area
# MAGIC - Step2: find out the crime event w.r.t category and time (hour) from the result of step 1  
# MAGIC - Step3: Give advice to distribute the police based on your analysis results. 

# COMMAND ----------

# MAGIC %md 
# MAGIC ###### Step 1: Find out the top-3 danger area

# COMMAND ----------

# DBTITLE 1,Spark dataframe based solution for Part 6 Step 1
df_6_s1 = df_crime.groupby('area_name').count().orderBy('count',ascending = False)
display(df_6_s1)

# COMMAND ----------

top3_danger = df_crime.groupby('area_name').count().orderBy('count',ascending = False).head(3)
top3_danger_area = [top3_danger[i][0] for i in range(3)]
top3_danger_area

# COMMAND ----------

# DBTITLE 1,Spark SQL based solution for Part 6 Step 1
sql_6_s1 = spark.sql( """
                       SELECT area_name, 
                              COUNT(*) as Count
                       FROM la_crime
                       GROUP BY 1
                       ORDER BY 2 DESC
                       LIMIT 3 
                       """ )
display(sql_6_s1)

# COMMAND ----------

# MAGIC %md 
# MAGIC ###### Step 2: Find out the crime event w.r.t category and time (hour) from the result of step 1 

# COMMAND ----------

df_6_s2 = df_crime.filter(df_crime.area_name.isin('77th Street', 'Central', 'Pacific')).groupby('crime_desc','Hour').count().orderBy('crime_desc','Hour')
display(df_6_s2)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Part 6_Travel Suggestion
# MAGIC 
# MAGIC 1. According to step1, the three most dangerous areas are 77th Street, Central and Pacific.
# MAGIC 2. We can see from the picture above that among the top three dangerous areas, the crime rate around 5 am is the lowest, and the high incidence of crime rate is around 12pm and 18pm, especially pay attention to assault, so I recommend to increase police patrol during that periods.

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Part 7 question
# MAGIC For different type of crime, find the percentage of status.

# COMMAND ----------

# MAGIC %sql select distinct(status_desc) as resolve from la_crime

# COMMAND ----------

import pyspark.sql.functions as f
from pyspark.sql.window import Window

resolution_func = udf(lambda x: x != 'Invest Cont')

df_7 = df_crime.withColumn('IsResolution', resolution_func(f.col('status_desc')))
df_7 = df_7.groupBy('crime_desc', 'status_desc', 'IsResolution').count().withColumnRenamed('count', 'resolved').orderBy('crime_desc')

df_7 = df_7.withColumn('total', f.sum('resolved').over(Window.partitionBy('crime_desc')))\
             .withColumn('percentage%', f.col('resolved')*100/f.col('total'))\
             .filter(df_7.IsResolution == True).orderBy('total', ascending=False)
display(df_7)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conclusion. 

# COMMAND ----------

# MAGIC %md
# MAGIC By analyzing crime data in Los Angeles using Spark SQL, I tried to give basic hints to the travellers and policemen about a big picture of when and how the number of crime would change in this city. I used mainly the DataFrame Structure with the help of SQL API to complete this task. I learned that the number of crime usually follows a trend with can be tracked, such as that the number of crime typically increases at noon, but this trend can only be captured by analyzing a huge amount of data, and Spark SQL is an ideal tool to handle this kind of big data. 
