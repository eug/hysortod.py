# HySortOD
<p align="center">
    <img width="150" src="hysortod-logo.svg"/>
</p>
Outlier Detection with Sorted Hypercubes. Java version is available in <a href="https://github.com/eug/hysortod.java">hysortod.java</a>.

### Install

```sh
pip install hysortod
```

### Example

```python
import pandas as pd
from hysortod import HySortOD

df = pd.read_csv("datasets/breastw.csv")
X = df.drop(columns='class')
y = df['class']

hysortod = HySortOD()
hysortod.fit(X)
print(hysortod.score(X, y))
```

### Reference
Eugenio F. Cabral and Robson L. F. Cordeiro. 2020. Fast and Scalable Outlier Detection with Sorted Hypercubes. In Proceedings of the 29th ACM International Conference on Information and Knowledge Management (CIKM'20), October 19–23, 2020. Virtual Event, Ireland. ACM, New York, NY, USA, 10 pages.
