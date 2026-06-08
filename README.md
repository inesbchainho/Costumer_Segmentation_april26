# machine_learning_feb2026
Project for Machine Learning II course | Group 9
Authors: Inês Chainho & Pedro Ambar
Dataset: Customer Segmentation Dataset (33038 observations)
Information: Demographics, spending behavior and purchasing history per customer
Problem: Unsupervised Learning (K-Means, HDBSCAN, Association Rules, Visualization through PCA/UMAP)

Variables — customer_info:
x   Column                                     Dtype     Variable Type              Meaning
0   customer_id                                uint16    categorical(nominal)       Unique identifier of the customer.
1   customer_name                              str       categorical(nominal)       Name of the customer. REMOVED
2   customer_gender                            uint8     categorical(nominal)       Gender of the customer (0, 1). REMOVED
3   customer_birthdate                         float16   categorical(nominal)       Birth date of the customer. REMOVED
4   kids_home                                  uint8     quantitative(discrete)     Number of kids at home.
5   teens_home                                 uint8     quantitative(discrete)     Number of teens at home.
6   number_complaints                          uint8     quantitative(discrete)     Number of formal complaints made by the customer.
7   distinct_stores_visited                    uint8     quantitative(discrete)     Number of distinct stores visited by the customer.
8   lifetime_spend_groceries                   float32   quantitative(continuous)   Total lifetime spend on groceries.
9   lifetime_spend_electronics                 float32   quantitative(continuous)   Total lifetime spend on electronics.
10  lifetime_spend_vegetables                  float32   quantitative(continuous)   Total lifetime spend on vegetables.
11  lifetime_spend_nonalcohol_drinks           float32   quantitative(continuous)   Total lifetime spend on non-alcoholic drinks.
12  lifetime_spend_alcohol_drinks              float32   quantitative(continuous)   Total lifetime spend on alcoholic drinks.
13  lifetime_spend_meat                        float32   quantitative(continuous)   Total lifetime spend on meat.
14  lifetime_spend_fish                        float32   quantitative(continuous)   Total lifetime spend on fish.
15  lifetime_spend_hygiene                     float32   quantitative(continuous)   Total lifetime spend on hygiene products.
16  lifetime_spend_petfood                     float32   quantitative(continuous)   Total lifetime spend on pet food.
17  lifetime_spend_videogames                  float32   quantitative(continuous)   Total lifetime spend on video games.
18  lifetime_total_distinct_products           float16   quantitative(discrete)     Number of distinct products bought over lifetime.
19  percentage_of_products_bought_promotion    float32   quantitative(continuous)   Percentage of products bought under a promotion (0-1).
20  year_first_transaction                     uint16    quantitative(discrete)     Year of the customer's first transaction of the customer. REMOVED
21  loyalty_card_number                        uint16    categorical(nominal)       Loyalty card number of the customer (0, 1). REMOVED
22  latitude                                   float32   quantitative(continuous)   Approximate latitude of the customer's home (<1km range).
23  longitude                                  float32   quantitative(continuous)   Approximate longitude of the customer's home (<1km range).
24  typical_hour                               uint8     quantitative(discrete)     Typical hour of the day when the customer visits the store.
25  degree_num                                 uint8     quantitative(discrete)     Maximum level of education (0-"None", 1-"Bsc", 2-"Msc", 3-"Phd").
26  is_female                                  uint8     quantitative(discrete)     If the customer is female (0, 1).
27  age                                        uint8     quantitative(discrete)     Age of the customer when the analysis was made (June 2026).
28  tenure                                     uint8     quantitative(discrete)     How many years ago was the customers' first transaction.
29  has_loyalty_card                           uint8     quantitative(discrete)     If the customer has a loyalty card (0, 1).
