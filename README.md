# Mori-Zwanzig-Formalism

# Code by: Louai Al Jabi

### Project  Overview
* The goal of the project is to calculate the diffusion constant using three different methods. Through Bootstrapping, the three methods will use a resampled portion of samples to measure said diffusion constant among other variables (Auto Correlation Functions and memory kernal values (k)). I recieved a version of the code that was able to calculate the diffusion constants without saving the other measured variables. Furthermore, the process was running using only one processor core, which considering the nature of the Mori-Zwanzig method, will take a long time to do a high number of bootstraps. Using multiprocessing, the process saw a speed increase of about %200 when running 1280 bootstraps. I added further optimizations to the code including functions to clean the produced data and present it properly. I also created over 10 graphs to show the results and study them for future alteration.

### My Changes and Improvements
* I studied the code and decided to introduce Multiprocessing into the process. This action came with many obstacles, mainly, understanding how multiprocessing works and which library would best suit the code we have. I settled on Concurrent Futures for its speed, logic, and useful utilities provided in the library. 
* I first used multiprocessing to speed up calculating Auto Correlation Functions. Three of them took 21.5mins to calculate. On the other hand, assigning each function to a core gave an expected 3x speed up at 7.2mins total. 
* After experimenting for three weeks, I have come to understand the way multiprocessing works and how I can use it to obtain our goal. First obstacle showed itself in the shape of process independancy, which means processes don't share a memory, however, they work mostly indepentantly. To remedy this, I wrote multiple functions that would generate the sample pool and split it accordangly between cores, which solved the problem.
* Also, I ran into a problem where some processes will end before the other and it would cause the pool to break due to the pending status. For that, I implimented the "as_completed" utility provided by the library.
* Unfortunately, my solutions lead to a new obstacle that I was aware off beforehand, however, I was told it wouldn't be a big deal. The problem was that due to the "as_completed" nature of the code, the data I recieved wasn't in order, or in other words, it didn't follow the order of the samples. 
* After deciding that it would benifit the goal to know from which sample was a specific diffusion constant calculated from, I had to go back and change the functions I wrote to accompany for this. I assigned an ID to each sample and integrated that ID into the final table to indicate that it was used to calculate all the diffusion constants and the other variables.
* The next step was to graph the data. I made 13 graphs each presenting unique aspects of this data. Each graph is color coded, labeled, and is measured with a 95% confidence interval.
* The final version of this code presents a DataFrame containing all the diffusion functions as well as other important variables along side the sampleID of the sample they were calculated from. Additionally, informative graphs that presents this data in a clear, cohecive way.

### Instructions
*
