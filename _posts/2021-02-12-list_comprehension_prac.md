---
title: "[DL 101] List comprehension"
date: 2021-02-12 14:000 -0400
excerpt: "Let's get familiar with list comprehension, instead of for loops"
author : 오승미
categories :
  - deep-learning
tags :
  - List-comprehension
  - python


---

# List comprehension

​	Instead of for loops, we should use list comprehension, equivalently,

```python
[expression for item in iterable if condition == True]
```

​	Why? so simple. Let's just assume that we want to print out all the letters in 'google' as a list.

​	If we use for loop,

```python
a = []

for letter in "google":
  a.append(letter)
```

​	On the other hand, with the list comprehension,

```python
a = [letter for letter in "google"]
```

​	As we can see clearly, the latter is simpler and it is definitely powerful when the function gets more complex. Thus, we should be accustomed to the list comprehension.

​	Let's practice with some examples.     

    l = [22, 13, 45, 50, 98, 69, 43, 44, 1]
    j = [12, 42, 52, 13, 24, 53, 94, 24, 25]
    k = [['A', 'B', 'C'] ,['D', 'E', 'F'], ['G', 'H', 'I']]
1. Subtract elements smaller than 30 in l  

   ```python
   [x for x in l if x < 30]
   ```

   the output is,

   ```python
    [22, 13, 1]
   ```



2. Tell if an element is smaller than 30 in l l

   ```python
   [x < 30 for x in l]
   ```

   the output is,

   ```python
    [True, True, False, False, False, False, False, False, True]
   ```



3. Extract elements in l which are bigger than those in j element-wisely

   ```python
   [x for i, x in enumerate(l) if x > j[i]]
   ```

   the output is,

   ```python
     [22, 50, 98, 69, 44]
   ```



4. Return element+10 if it is bigger than j element-wisely, otherwise return element-10

   ```python
   [x+10 if x > j[i] else j[i]-10 for i, x in enumerate(l)]
   ```

   the output is,

   ```python
   [32, 32, 42, 60, 108, 79, 84, 54, 15]
   ```



5. Change the middle value of each nested list in k to lower case

   ```python
   [[val.lower() if i ==1 else val for i, val in enumerate(x)] for x in k]
   ```

   the output is,

   ```python
     [['A', 'b', 'C'] ['D', 'e', 'F'], ['G', 'h', 'I']]
   ```
