from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, col

# Step 1: Initialize Spark session
spark = SparkSession.builder.appName("WordCount").getOrCreate()

# Step 2: Load text file into a DataFrame
file_path = "/home/ran/apache-playground/spark-playground/afeka-big-data-course/src/homework_1/romeo-juliet.txt"  # Adjust this path as needed
text_df = spark.read.text(file_path)

# Step 3: Process and count words
# Split each line into words, then explode to have one word per row
words_df = text_df.select(explode(split(col("value"), "\\s+")).alias("word"))

# Filter out any empty strings
words_df = words_df.filter(words_df.word != "")

# Step 4: Count each word
word_count_df = words_df.groupBy("word").count()

# Step 5: Sort and display
sorted_word_count_df = word_count_df.orderBy(col("count").desc())
sorted_word_count_df.show(20)  # Display top 20 words

# Optional: Save results to a file
sorted_word_count_df.write.csv(
    "/home/ran/apache-playground/spark-playground/afeka-big-data-course/src/homework_1/output/word_counts.csv"
)

# Stop the Spark session
spark.stop()
