# Text Clustering Report

- Records: 241
- Selected clusters: 4
- Stop-word removal: True
- Lemmatization: True
- Word2Vec: vector_size=100, window=5, min_count=1, epochs=100

## KMeans evaluation

| k  | inertia             | silhouette          |
| -- | ------------------- | ------------------- |
| 2  | 0.47472047805786133 | 0.2244788259267807  |
| 3  | 0.44318240880966187 | 0.11477430164813995 |
| 4  | 0.4290209412574768  | 0.08809278160333633 |
| 5  | 0.4192458987236023  | 0.09637092798948288 |
| 6  | 0.41169747710227966 | 0.09666789323091507 |
| 7  | 0.4022925794124603  | 0.05959762632846832 |
| 8  | 0.39258357882499695 | 0.0977737307548523  |
| 9  | 0.3886566162109375  | 0.09643708169460297 |
| 10 | 0.38083556294441223 | 0.10294191539287567 |

## Cluster analysis

### Cluster 0 (46 records, 19.1%)
- Representatives: password management | business management | wealth management | employee management | brand management
- Common terms: management, online, store, home, shopping, education, security, car, software, insurance
- Distinguishing terms: management, store, shopping, online, home, education, car, security, rental, software
- Interpretation: Cluster 0: records associated with terms management, store, shopping.; cluster separation may be weak.

### Cluster 1 (54 records, 22.4%)
- Representatives: business consulting | small business | business banking | investment banking | business loans
- Common terms: business, marketing, consulting, security, digital, banking, travel, booking, cloud, loan
- Distinguishing terms: marketing, booking, business, digital, consulting, banking, travel, security, loan, information
- Interpretation: Cluster 1: records associated with terms marketing, booking, business.; cluster separation may be weak.

### Cluster 2 (44 records, 18.3%)
- Representatives: business services | telemedicine services | construction services | airport services | plumbing services
- Common terms: service, real, estate, cloud, banking, financial, insurance, loan, mortgage, payment
- Distinguishing terms: service, real, estate, plumbing, pharmacy, mortgage, nutrition, recruitment, resume, telemedicine
- Interpretation: Cluster 2: records associated with terms service, real, estate..

### Cluster 3 (97 records, 40.2%)
- Representatives: online restaurant | customer reviews | investment planning | data science courses | business education
- Common terms: online, education, learning, planning, course, car, delivery, software, data, health
- Distinguishing terms: planning, learning, delivery, course, technology, review, data, health, car, sharing
- Interpretation: Cluster 3: records associated with terms planning, learning, delivery.; cluster separation may be weak.
