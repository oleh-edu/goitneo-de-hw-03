from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession 
from pyspark.sql.functions import col, sum as spark_sum, round, lit

# Configure Spark
conf = SparkConf().setAppName("Data Engineering: HW-03").setMaster("local[*]")
sc = SparkContext(conf=conf)

# Set log level to ERROR
sc.setLogLevel("ERROR")

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("Data Engineering: HW-03") \
    .getOrCreate()

# File paths
dataset_dir = "./dataset"
users_file_path = f"{dataset_dir}/users.csv"
purchases_file_path = f"{dataset_dir}/purchases.csv"
products_file_path = f"{dataset_dir}/products.csv"

# 1. Load CSV files
print("[*] Loading data...")
users_df = spark.read.csv(users_file_path, header=True, inferSchema=True)
purchases_df = spark.read.csv(purchases_file_path, header=True, inferSchema=True)
products_df = spark.read.csv(products_file_path, header=True, inferSchema=True)
print("[*] Data loaded successfully.")

# 2. Clean data by removing rows with missing values
print("[*] Cleaning data...")
users_df = users_df.dropna()
purchases_df = purchases_df.dropna()
products_df = products_df.dropna()
print("[*] Data cleaning completed.")

# Join data
print("[*] Joining datasets...")
# Join purchases with products by product_id
purchases_products_df = purchases_df.join(products_df, "product_id")
# Join the result with users by user_id
full_data_df = purchases_products_df.join(users_df, "user_id")
print("[*] Data joined successfully.")

# 3. Calculate the total purchase amount for each product category
print("[*] Calculating total purchases by category...")
category_sales = full_data_df.groupBy("category").agg(
    round(spark_sum(col("quantity") * col("price")), 2).alias("total_sales")
)
category_sales.show()

# 4. Calculate purchase amounts by category for the 18-25 age group
print("[*] Calculating purchases for the 18-25 age group...")
age_filtered_data = full_data_df.filter((col("age") >= 18) & (col("age") <= 25))
category_sales_18_25 = age_filtered_data.groupBy("category").agg(
    round(spark_sum(col("quantity") * col("price")), 2).alias("total_sales_18_25")
)
category_sales_18_25.show()

# 5. Calculate the share of purchases by category relative to total expenditures for the 18-25 age group
print("[*] Calculating category percentages for the 18-25 age group...")
total_sales_18_25 = category_sales_18_25.agg(spark_sum("total_sales_18_25").alias("grand_total_18_25")).collect()[0]["grand_total_18_25"]
category_percentage_18_25 = category_sales_18_25.withColumn(
    "percentage", round((col("total_sales_18_25") / lit(total_sales_18_25)) * 100, 2)
)
category_percentage_18_25.show()

# 6. Select the top 3 product categories with the highest percentage of spending for the 18-25 age group
print("[*] Selecting top 3 categories by spending percentage for the 18-25 age group...")
top_3_categories = category_percentage_18_25.orderBy(col("percentage").desc()).limit(3)
top_3_categories.show()
