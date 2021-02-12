---
Title: Fundamentals of the Recommendation System
date: 2021-02-09 14:00:00-0400
Author: 오승미
categories : 
	- recommendation system
	- collaborative filtering
	- content-based

---

# Fundamentals of the Recommendation System

Based on the *[Advanced Machine Learning with TensorFlow on Google Cloud Platform] (https://www.coursera.org/learn/recommendation-models-gcp)*, this article looks up fundamental recommendation system models, content-based, collaborative-filtering and knowledge-based.

-----

# Content-Based Model

#### 	*Using attributes of items to recommend new item to an user*



## How it works?

### 	Step 1. User-Item Rating Matrix

|   🙎‍♀️ / Movie   | Rating |
| :------------: | :----: |
|    Minions     |   7    |
| A Star is Born |   4    |
|    Aladdin     |   10   |

​	Let's just assume that we are given the table above. In this table, the user🙎‍♀️ gave 7-star ratings to "Minions", four-star ratings to "A Star is Born" and 10-ratings to "Aladdin" but we do not have information about other movies.

### 	Step 2. Item Feature Matrix

|  Movie\Genre   | Fantasy | Action | Cartoon | Drama | Comedy |
| :------------: | :-----: | :----: | :-----: | :---: | :----: |
|    Minions     |    0    |   0    |    1    |   0   |   1    |
| A Star is Born |    0    |   0    |    0    |   1   |   0    |
|    Aladdin     |    1    |   0    |    0    |   0   |   1    |

​	For those rated movies, we now build an item feature matrix of genres. We can also consider themes, actors/directors, professional ratings, movie summary text, stills from movie, movie trailer as features. In the table above, "Minions" belongs to Cartoon and Comedy categories, "A Star is Born" is Drama and "Aladdin" is Fantasy and Comedy movie.

### 	Step 3. User Feature Vector

​	Then apply dot-product to the user-item rating matrix and the item feature matrix and sum up the matrix by column as below.

| Fantasy | Action | Cartoon | Drama | Comedy |
| :-----: | :----: | :-----: | :---: | :----: |
|   10    |   0    |    7    |   4   |   17   |

​	The above matrix is a five-dimensional embedded feature space that we use to represent movies. Normalizing the above matrix, now we get the **user feature vector**. 

| Fantasy | Action | Cartoon | Drama | Comedy |
| :-----: | :----: | :-----: | :---: | :----: |
|  0.26   |   0    |  0.18   | 0.11  |  0.45  |

​	The above vector is the user feature vector. Note that "0" for the Action genre does not mean that the user dislikes it because none of the movies he/she has previously rated contains the Action feature. 

### Step 4. User Rating Prediction

​	So, how can we include those concepts to engineer the content-based recommendation system? Now we have a new item feature matrix as below. Movies below can be both seen and unseen movies by the user (Simply we can just predict all ratings and drop off seen movies at the last step of recommendation). In this case, however, we only take care of four unseen movies.

|      Movie\Genre      | Fantasy | Action | Cartoon | Drama | Comedy |
| :-------------------: | :-----: | :----: | :-----: | :---: | :----: |
|     Harry Potter      |    1    |   1    |    0    |   0   |   0    |
| The Dark Knight Rises |    1    |   1    |    0    |   1   |   0    |
|      Incredible       |    0    |   1    |    1    |   0   |   1    |
|        Memento        |    0    |   0    |    0    |   1   |   0    |

​		Then multiply the user feature vector component-wisely to the new movie feature vector for each movie and then sum row-wise to compute the dot product. This gives us the dot product similarity between the user and each of those four movies.

| Harry Potter | The Dark Knight Rises | Incredible | Memento |
| :----------: | :-------------------: | :--------: | :-----: |
|     0.71     |         0.37          |    0.63    |  0.11   |

​		The above vector shows predicted ratings for each movie. Clearly, the higher the more likely (hopefully) the user likes the item. Thus, we should recommend "Harry Potter" and "Incredible" to this user but not other movies. Simple, isn't it?



______

# Collaborative-Filtering Model

#### 	*Using attributes of items to recommend new item to an user*



## Matrix Factorization

​	Assume that we have user-item interaction matrix of muliple users denoting whether the user watches or not.

| User\Movie | Harry Potter | Incredible | Shrek | Dark Knight Rises | Memento |
| :--------: | :----------: | :--------: | :---: | :---------------: | :-----: |
|     🙎‍♀️     |      O       |            |   O   |         O         |         |
|     🙎      |              |     O      |       |                   |    O    |
|    🙎🏾‍♂️     |      O       |     O      |   O   |                   |         |
|    🙍🏼‍♀️     |              |            |       |         O         |    O    |

​	As the more users and movies are collected, the user-item matrix gets more sparse; normally needs to be shrunk down to more attractable size through **matrix factorization**. The factorization splits this matrix into row factors and column factors that are essentially user and item embeddings. Let A to be the whole user-item interaction matrix, then it is decomposed into U (user embedding) and V (item embedding) like below (much smaller!). 
$$
A \approx U \times V^T
$$
​	Each user and item is a d-dimensional point within an embedding space. Embeddings can be learned from data;  *PCA, SVD, etc*. Compress data to find the best generalities to rely on, called latent factors. This saves space as long as the number of latent factors, k, is smaller than the harmonic mean of the number of users and items, (U\*V)/(2*(U+V)).  In CF, however, the problem is when we do not have information of new users, called **cold start**. In this case, we can use averages from the other users/items for fresh users/items until sufficient interactions are made.

------



# Summary

**pros and cons**

| Method |                        Content-Based                         |                    Collborative Filtering                    |                       Knowledge-Based                        |
| :----: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|  pros  | - No need for data about other users<br /> - Can recommend niche items | - No domain knowledge<br />- Serendipity<br />- Great starting point | - No interaction data needed<br />- Usually high-fidelity data from user self-reporting |
|  cons  |   - Need domain knowledge<br />- Only safe recommendations   |   - Cold start<br />- Sparsity<br />- No context features    | - Need user data<br />- Need to be careful with privacy concerns |



**structured/unstructured information that can be used**

|     Type     |                        Content-Based                         |                    Collborative Filtering                    |                       Knowledge-Based                        |
| :----------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|  structured  | - Genres<br />- Themes<br />- Actors/directors involved<br />- Professional ratings | - User ratings<br />- User views<br />- User wishlist/cart history<br />- User purchase/return history | - Demographic information<br />- Location/country/language<br />- Genre preferences<br />- Global filters |
| Unstructured | - Movie summary test<br />- Stills from moview<br />- Movie trailer<br />- Professional reviews | - User reviews<br />- User-answered questions<br />- User-submitted photos<br />- User-submitted videos |                  - User 'about me' snippets                  |