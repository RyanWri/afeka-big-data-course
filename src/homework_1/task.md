Lab1

The purpose of this Lab is to practice Basic Spark Functionality
1. Connect to the VLAb with the user linuxu / Linux1212 / Academic
2. Launch the Terminl  (ALT + CTRL + T). 
    • Start Spark with pyspark command. Create the Python3 Notebook
    • Type  sc and review the info printed. Repeat for spark. Stop the session, display spark. Restart the Notebook. Verify if the session exists (display spark).
    • Create collection data with elemlets [21, 27, 43, 40, 75, 61, 70, 8, 9, 100, 11]  and distribute it between the nodes; name it firstRDD.
        a. Display first element of your local collection data and distributed collection firstRDD. Try commands take, first.
        b. Create and display a new RDD secondRDD where each element is bigger by 123 of corresponding element of firstRDD.
        c. Find maximum element of firstRDD and minimum element of secondRDD.
        d. Create a new RDD thirdRDD with elements bigger than 150.

3. Create text file “data.txt” with a few lines. Create RDD of this text File and display it.

4. Download the file “Romeo and Juliet.txt” from the course site.
    • Count the number of lines and number of words in this file
    • Find and display the longest phrase in the file
    • Find ten most frequent words in the file
    • Find who is the mostly speaking character – Romeo (Rom.) or Juliet (Jul.) ? 


5. Use Spark to calculate approximation of integral based on following formula1 :
    Check your development for a = 0,   b = 10,   f(x) = x2 - 3
