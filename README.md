# Movie_Recommendation
Movie Recommendations 

```python
import pandas as pd
movies=pd.read_csv("movies.csv")
```


```python
movies.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama|Romance</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
    </tr>
  </tbody>
</table>
</div>




```python
import re

def clean_title(title):
    title = re.sub("[^a-zA-Z0-9 ]", "", title)
    return title

```


```python
movies["clean_title"] = movies["title"].apply(clean_title)
```


```python

movies

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>clean_title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>Toy Story 1995</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
      <td>Jumanji 1995</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
      <td>Grumpier Old Men 1995</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama|Romance</td>
      <td>Waiting to Exhale 1995</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
      <td>Father of the Bride Part II 1995</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>62418</th>
      <td>209157</td>
      <td>We (2018)</td>
      <td>Drama</td>
      <td>We 2018</td>
    </tr>
    <tr>
      <th>62419</th>
      <td>209159</td>
      <td>Window of the Soul (2001)</td>
      <td>Documentary</td>
      <td>Window of the Soul 2001</td>
    </tr>
    <tr>
      <th>62420</th>
      <td>209163</td>
      <td>Bad Poems (2018)</td>
      <td>Comedy|Drama</td>
      <td>Bad Poems 2018</td>
    </tr>
    <tr>
      <th>62421</th>
      <td>209169</td>
      <td>A Girl Thing (2001)</td>
      <td>(no genres listed)</td>
      <td>A Girl Thing 2001</td>
    </tr>
    <tr>
      <th>62422</th>
      <td>209171</td>
      <td>Women of Devil's Island (1962)</td>
      <td>Action|Adventure|Drama</td>
      <td>Women of Devils Island 1962</td>
    </tr>
  </tbody>
</table>
<p>62423 rows × 4 columns</p>
</div>




```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,2))

tfidf = vectorizer.fit_transform(movies["clean_title"])
```


```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def search(title):
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies.iloc[indices].iloc[::-1]
    
    return results
```


```python
import ipywidgets as widgets
from IPython.display import display

movie_input = widgets.Text(
    value='Toy Story',
    description='Movie Title:',
    disabled=False
)
movie_input
 movie_list = widgets.Output()

 def on_type(data):
     with movie_list:
         movie_list.clear_output()
         title = data["new"]
         if len(title) > 5:
             display(search(title))

 movie_input.observe(on_type, names='value')


 display(movie_input, movie_list)
```


      Cell In[2], line 10
        movie_list = widgets.Output()
        ^
    IndentationError: unexpected indent
    



```python
pip install ipywidgets
```

    Requirement already satisfied: ipywidgets in c:\users\kavya\anaconda3\lib\site-packages (7.8.1)Note: you may need to restart the kernel to use updated packages.
    
    Requirement already satisfied: comm>=0.1.3 in c:\users\kavya\anaconda3\lib\site-packages (from ipywidgets) (0.2.1)
    Requirement already satisfied: ipython-genutils~=0.2.0 in c:\users\kavya\anaconda3\lib\site-packages (from ipywidgets) (0.2.0)
    Requirement already satisfied: traitlets>=4.3.1 in c:\users\kavya\anaconda3\lib\site-packages (from ipywidgets) (5.14.3)
    Requirement already satisfied: widgetsnbextension~=3.6.6 in c:\users\kavya\anaconda3\lib\site-packages (from ipywidgets) (3.6.6)
    Requirement already satisfied: ipython>=4.0.0 in c:\users\kavya\anaconda3\lib\site-packages (from ipywidgets) (8.27.0)
    Requirement already satisfied: jupyterlab-widgets<3,>=1.0.0 in c:\users\kavya\anaconda3\lib\site-packages (from ipywidgets) (1.0.0)
    Requirement already satisfied: decorator in c:\users\kavya\anaconda3\lib\site-packages (from ipython>=4.0.0->ipywidgets) (5.1.1)
    Requirement already satisfied: jedi>=0.16 in c:\users\kavya\anaconda3\lib\site-packages (from ipython>=4.0.0->ipywidgets) (0.19.1)
    Requirement already satisfied: matplotlib-inline in c:\users\kavya\anaconda3\lib\site-packages (from ipython>=4.0.0->ipywidgets) (0.1.6)
    Requirement already satisfied: prompt-toolkit<3.1.0,>=3.0.41 in c:\users\kavya\anaconda3\lib\site-packages (from ipython>=4.0.0->ipywidgets) (3.0.43)
    Requirement already satisfied: pygments>=2.4.0 in c:\users\kavya\anaconda3\lib\site-packages (from ipython>=4.0.0->ipywidgets) (2.15.1)
    Requirement already satisfied: stack-data in c:\users\kavya\anaconda3\lib\site-packages (from ipython>=4.0.0->ipywidgets) (0.2.0)
    Requirement already satisfied: colorama in c:\users\kavya\anaconda3\lib\site-packages (from ipython>=4.0.0->ipywidgets) (0.4.6)
    Requirement already satisfied: notebook>=4.4.1 in c:\users\kavya\anaconda3\lib\site-packages (from widgetsnbextension~=3.6.6->ipywidgets) (7.2.2)
    Requirement already satisfied: parso<0.9.0,>=0.8.3 in c:\users\kavya\anaconda3\lib\site-packages (from jedi>=0.16->ipython>=4.0.0->ipywidgets) (0.8.3)
    Requirement already satisfied: jupyter-server<3,>=2.4.0 in c:\users\kavya\anaconda3\lib\site-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (2.14.1)
    Requirement already satisfied: jupyterlab-server<3,>=2.27.1 in c:\users\kavya\anaconda3\lib\site-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (2.27.3)
    Requirement already satisfied: jupyterlab<4.3,>=4.2.0 in c:\users\kavya\anaconda3\lib\site-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (4.2.5)
    Requirement already satisfied: notebook-shim<0.3,>=0.2 in c:\users\kavya\anaconda3\lib\site-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (0.2.3)
    Requirement already satisfied: tornado>=6.2.0 in c:\users\kavya\anaconda3\lib\site-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (6.4.1)
    Requirement already satisfied: wcwidth in c:\users\kavya\anaconda3\lib\site-packages (from prompt-toolkit<3.1.0,>=3.0.41->ipython>=4.0.0->ipywidgets) (0.2.5)
    Requirement already satisfied: executing in c:\users\kavya\anaconda3\lib\site-packages (from stack-data->ipython>=4.0.0->ipywidgets) (0.8.3)
    Requirement already satisfied: asttokens in c:\users\kavya\anaconda3\lib\site-packages (from stack-data->ipython>=4.0.0->ipywidgets) (2.0.5)
    Requirement already satisfied: pure-eval in c:\users\kavya\anaconda3\lib\site-packages (from stack-data->ipython>=4.0.0->ipywidgets) (0.2.2)
    Requirement already satisfied: anyio>=3.1.0 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (4.2.0)
    Requirement already satisfied: argon2-cffi>=21.1 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (21.3.0)
    Requirement already satisfied: jinja2>=3.0.3 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (3.1.4)
    Requirement already satisfied: jupyter-client>=7.4.4 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (8.6.0)
    Requirement already satisfied: jupyter-core!=5.0.*,>=4.12 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (5.7.2)
    Requirement already satisfied: jupyter-events>=0.9.0 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (0.10.0)
    Requirement already satisfied: jupyter-server-terminals>=0.4.4 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (0.4.4)
    Requirement already satisfied: nbconvert>=6.4.4 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (7.16.4)
    Requirement already satisfied: nbformat>=5.3.0 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (5.10.4)
    Requirement already satisfied: overrides>=5.0 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (7.4.0)
    Requirement already satisfied: packaging>=22.0 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (24.1)
    Requirement already satisfied: prometheus-client>=0.9 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (0.14.1)
    Requirement already satisfied: pywinpty>=2.0.1 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (2.0.10)
    Requirement already satisfied: pyzmq>=24 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (25.1.2)
    Requirement already satisfied: send2trash>=1.8.2 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (1.8.2)
    Requirement already satisfied: terminado>=0.8.3 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (0.17.1)
    Requirement already satisfied: websocket-client>=1.7 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (1.8.0)
    Requirement already satisfied: async-lru>=1.0.0 in c:\users\kavya\anaconda3\lib\site-packages (from jupyterlab<4.3,>=4.2.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (2.0.4)
    Requirement already satisfied: httpx>=0.25.0 in c:\users\kavya\anaconda3\lib\site-packages (from jupyterlab<4.3,>=4.2.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (0.27.0)
    Requirement already satisfied: ipykernel>=6.5.0 in c:\users\kavya\anaconda3\lib\site-packages (from jupyterlab<4.3,>=4.2.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (6.28.0)
    Requirement already satisfied: jupyter-lsp>=2.0.0 in c:\users\kavya\anaconda3\lib\site-packages (from jupyterlab<4.3,>=4.2.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (2.2.0)
    Requirement already satisfied: setuptools>=40.1.0 in c:\users\kavya\anaconda3\lib\site-packages (from jupyterlab<4.3,>=4.2.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (75.1.0)
    Requirement already satisfied: babel>=2.10 in c:\users\kavya\anaconda3\lib\site-packages (from jupyterlab-server<3,>=2.27.1->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (2.11.0)
    Requirement already satisfied: json5>=0.9.0 in c:\users\kavya\anaconda3\lib\site-packages (from jupyterlab-server<3,>=2.27.1->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (0.9.6)
    Requirement already satisfied: jsonschema>=4.18.0 in c:\users\kavya\anaconda3\lib\site-packages (from jupyterlab-server<3,>=2.27.1->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (4.23.0)
    Requirement already satisfied: requests>=2.31 in c:\users\kavya\anaconda3\lib\site-packages (from jupyterlab-server<3,>=2.27.1->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (2.32.3)
    Requirement already satisfied: six in c:\users\kavya\anaconda3\lib\site-packages (from asttokens->stack-data->ipython>=4.0.0->ipywidgets) (1.16.0)
    Requirement already satisfied: idna>=2.8 in c:\users\kavya\anaconda3\lib\site-packages (from anyio>=3.1.0->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (3.7)
    Requirement already satisfied: sniffio>=1.1 in c:\users\kavya\anaconda3\lib\site-packages (from anyio>=3.1.0->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (1.3.0)
    Requirement already satisfied: argon2-cffi-bindings in c:\users\kavya\anaconda3\lib\site-packages (from argon2-cffi>=21.1->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (21.2.0)
    Requirement already satisfied: pytz>=2015.7 in c:\users\kavya\anaconda3\lib\site-packages (from babel>=2.10->jupyterlab-server<3,>=2.27.1->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (2024.1)
    Requirement already satisfied: certifi in c:\users\kavya\anaconda3\lib\site-packages (from httpx>=0.25.0->jupyterlab<4.3,>=4.2.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (2024.12.14)
    Requirement already satisfied: httpcore==1.* in c:\users\kavya\anaconda3\lib\site-packages (from httpx>=0.25.0->jupyterlab<4.3,>=4.2.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (1.0.2)
    Requirement already satisfied: h11<0.15,>=0.13 in c:\users\kavya\anaconda3\lib\site-packages (from httpcore==1.*->httpx>=0.25.0->jupyterlab<4.3,>=4.2.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (0.14.0)
    Requirement already satisfied: debugpy>=1.6.5 in c:\users\kavya\anaconda3\lib\site-packages (from ipykernel>=6.5.0->jupyterlab<4.3,>=4.2.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (1.6.7)
    Requirement already satisfied: nest-asyncio in c:\users\kavya\anaconda3\lib\site-packages (from ipykernel>=6.5.0->jupyterlab<4.3,>=4.2.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (1.6.0)
    Requirement already satisfied: psutil in c:\users\kavya\anaconda3\lib\site-packages (from ipykernel>=6.5.0->jupyterlab<4.3,>=4.2.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (5.9.0)
    Requirement already satisfied: MarkupSafe>=2.0 in c:\users\kavya\anaconda3\lib\site-packages (from jinja2>=3.0.3->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (2.1.3)
    Requirement already satisfied: attrs>=22.2.0 in c:\users\kavya\anaconda3\lib\site-packages (from jsonschema>=4.18.0->jupyterlab-server<3,>=2.27.1->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (23.1.0)
    Requirement already satisfied: jsonschema-specifications>=2023.03.6 in c:\users\kavya\anaconda3\lib\site-packages (from jsonschema>=4.18.0->jupyterlab-server<3,>=2.27.1->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (2023.7.1)
    Requirement already satisfied: referencing>=0.28.4 in c:\users\kavya\anaconda3\lib\site-packages (from jsonschema>=4.18.0->jupyterlab-server<3,>=2.27.1->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (0.30.2)
    Requirement already satisfied: rpds-py>=0.7.1 in c:\users\kavya\anaconda3\lib\site-packages (from jsonschema>=4.18.0->jupyterlab-server<3,>=2.27.1->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (0.10.6)
    Requirement already satisfied: python-dateutil>=2.8.2 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-client>=7.4.4->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (2.9.0.post0)
    Requirement already satisfied: platformdirs>=2.5 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-core!=5.0.*,>=4.12->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (3.10.0)
    Requirement already satisfied: pywin32>=300 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-core!=5.0.*,>=4.12->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (305.1)
    Requirement already satisfied: python-json-logger>=2.0.4 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-events>=0.9.0->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (2.0.7)
    Requirement already satisfied: pyyaml>=5.3 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-events>=0.9.0->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (6.0.1)
    Requirement already satisfied: rfc3339-validator in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-events>=0.9.0->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (0.1.4)
    Requirement already satisfied: rfc3986-validator>=0.1.1 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-events>=0.9.0->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (0.1.1)
    Requirement already satisfied: beautifulsoup4 in c:\users\kavya\anaconda3\lib\site-packages (from nbconvert>=6.4.4->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (4.12.3)
    Requirement already satisfied: bleach!=5.0.0 in c:\users\kavya\anaconda3\lib\site-packages (from nbconvert>=6.4.4->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (4.1.0)
    Requirement already satisfied: defusedxml in c:\users\kavya\anaconda3\lib\site-packages (from nbconvert>=6.4.4->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (0.7.1)
    Requirement already satisfied: jupyterlab-pygments in c:\users\kavya\anaconda3\lib\site-packages (from nbconvert>=6.4.4->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (0.1.2)
    Requirement already satisfied: mistune<4,>=2.0.3 in c:\users\kavya\anaconda3\lib\site-packages (from nbconvert>=6.4.4->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (2.0.4)
    Requirement already satisfied: nbclient>=0.5.0 in c:\users\kavya\anaconda3\lib\site-packages (from nbconvert>=6.4.4->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (0.8.0)
    Requirement already satisfied: pandocfilters>=1.4.1 in c:\users\kavya\anaconda3\lib\site-packages (from nbconvert>=6.4.4->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (1.5.0)
    Requirement already satisfied: tinycss2 in c:\users\kavya\anaconda3\lib\site-packages (from nbconvert>=6.4.4->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (1.2.1)
    Requirement already satisfied: fastjsonschema>=2.15 in c:\users\kavya\anaconda3\lib\site-packages (from nbformat>=5.3.0->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (2.16.2)
    Requirement already satisfied: charset-normalizer<4,>=2 in c:\users\kavya\anaconda3\lib\site-packages (from requests>=2.31->jupyterlab-server<3,>=2.27.1->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (3.3.2)
    Requirement already satisfied: urllib3<3,>=1.21.1 in c:\users\kavya\anaconda3\lib\site-packages (from requests>=2.31->jupyterlab-server<3,>=2.27.1->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (2.2.3)
    Requirement already satisfied: webencodings in c:\users\kavya\anaconda3\lib\site-packages (from bleach!=5.0.0->nbconvert>=6.4.4->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (0.5.1)
    Collecting fqdn (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.9.0->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets)
      Using cached fqdn-1.5.1-py3-none-any.whl.metadata (1.4 kB)
    Collecting isoduration (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.9.0->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets)
      Using cached isoduration-20.11.0-py3-none-any.whl.metadata (5.7 kB)
    Requirement already satisfied: jsonpointer>1.13 in c:\users\kavya\anaconda3\lib\site-packages (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.9.0->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (2.1)
    Collecting uri-template (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.9.0->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets)
      Using cached uri_template-1.3.0-py3-none-any.whl.metadata (8.8 kB)
    Collecting webcolors>=24.6.0 (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.9.0->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets)
      Using cached webcolors-24.11.1-py3-none-any.whl.metadata (2.2 kB)
    Requirement already satisfied: cffi>=1.0.1 in c:\users\kavya\anaconda3\lib\site-packages (from argon2-cffi-bindings->argon2-cffi>=21.1->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (1.17.1)
    Requirement already satisfied: soupsieve>1.2 in c:\users\kavya\anaconda3\lib\site-packages (from beautifulsoup4->nbconvert>=6.4.4->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (2.5)
    Requirement already satisfied: pycparser in c:\users\kavya\anaconda3\lib\site-packages (from cffi>=1.0.1->argon2-cffi-bindings->argon2-cffi>=21.1->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (2.21)
    Requirement already satisfied: arrow>=0.15.0 in c:\users\kavya\anaconda3\lib\site-packages (from isoduration->jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.9.0->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (1.2.3)
    Using cached webcolors-24.11.1-py3-none-any.whl (14 kB)
    Using cached fqdn-1.5.1-py3-none-any.whl (9.1 kB)
    Using cached isoduration-20.11.0-py3-none-any.whl (11 kB)
    Using cached uri_template-1.3.0-py3-none-any.whl (11 kB)
    Installing collected packages: webcolors, uri-template, fqdn, isoduration
    Successfully installed fqdn-1.5.1 isoduration-20.11.0 uri-template-1.3.0 webcolors-24.11.1
    


```python
import ipywidgets as widgets
from IPython.display import display

movie_input = widgets.Text(
    value='Toy Story',
    description='Movie Title:',
    disabled=False
)
movie_input
```




    Text(value='Toy Story', description='Movie Title:')




```python
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```


      Cell In[25], line 1
        jupyter labextension install @jupyter-widgets/jupyterlab-manager
                ^
    SyntaxError: invalid syntax
    



```python
pip install ipywidgets
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```


      Cell In[27], line 1
        pip install ipywidgets
            ^
    SyntaxError: invalid syntax
    



```python
pip install ipywidgets

```

    Requirement already satisfied: ipywidgets in c:\users\kavya\anaconda3\lib\site-packages (7.8.1)
    Requirement already satisfied: comm>=0.1.3 in c:\users\kavya\anaconda3\lib\site-packages (from ipywidgets) (0.2.1)
    Requirement already satisfied: ipython-genutils~=0.2.0 in c:\users\kavya\anaconda3\lib\site-packages (from ipywidgets) (0.2.0)
    Requirement already satisfied: traitlets>=4.3.1 in c:\users\kavya\anaconda3\lib\site-packages (from ipywidgets) (5.14.3)
    Requirement already satisfied: widgetsnbextension~=3.6.6 in c:\users\kavya\anaconda3\lib\site-packages (from ipywidgets) (3.6.6)
    Requirement already satisfied: ipython>=4.0.0 in c:\users\kavya\anaconda3\lib\site-packages (from ipywidgets) (8.27.0)
    Requirement already satisfied: jupyterlab-widgets<3,>=1.0.0 in c:\users\kavya\anaconda3\lib\site-packages (from ipywidgets) (1.0.0)
    Requirement already satisfied: decorator in c:\users\kavya\anaconda3\lib\site-packages (from ipython>=4.0.0->ipywidgets) (5.1.1)
    Requirement already satisfied: jedi>=0.16 in c:\users\kavya\anaconda3\lib\site-packages (from ipython>=4.0.0->ipywidgets) (0.19.1)
    Requirement already satisfied: matplotlib-inline in c:\users\kavya\anaconda3\lib\site-packages (from ipython>=4.0.0->ipywidgets) (0.1.6)
    Requirement already satisfied: prompt-toolkit<3.1.0,>=3.0.41 in c:\users\kavya\anaconda3\lib\site-packages (from ipython>=4.0.0->ipywidgets) (3.0.43)
    Requirement already satisfied: pygments>=2.4.0 in c:\users\kavya\anaconda3\lib\site-packages (from ipython>=4.0.0->ipywidgets) (2.15.1)
    Requirement already satisfied: stack-data in c:\users\kavya\anaconda3\lib\site-packages (from ipython>=4.0.0->ipywidgets) (0.2.0)
    Requirement already satisfied: colorama in c:\users\kavya\anaconda3\lib\site-packages (from ipython>=4.0.0->ipywidgets) (0.4.6)
    Requirement already satisfied: notebook>=4.4.1 in c:\users\kavya\anaconda3\lib\site-packages (from widgetsnbextension~=3.6.6->ipywidgets) (7.2.2)
    Requirement already satisfied: parso<0.9.0,>=0.8.3 in c:\users\kavya\anaconda3\lib\site-packages (from jedi>=0.16->ipython>=4.0.0->ipywidgets) (0.8.3)
    Requirement already satisfied: jupyter-server<3,>=2.4.0 in c:\users\kavya\anaconda3\lib\site-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (2.14.1)
    Requirement already satisfied: jupyterlab-server<3,>=2.27.1 in c:\users\kavya\anaconda3\lib\site-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (2.27.3)
    Requirement already satisfied: jupyterlab<4.3,>=4.2.0 in c:\users\kavya\anaconda3\lib\site-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (4.2.5)
    Requirement already satisfied: notebook-shim<0.3,>=0.2 in c:\users\kavya\anaconda3\lib\site-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (0.2.3)
    Requirement already satisfied: tornado>=6.2.0 in c:\users\kavya\anaconda3\lib\site-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (6.4.1)
    Requirement already satisfied: wcwidth in c:\users\kavya\anaconda3\lib\site-packages (from prompt-toolkit<3.1.0,>=3.0.41->ipython>=4.0.0->ipywidgets) (0.2.5)
    Requirement already satisfied: executing in c:\users\kavya\anaconda3\lib\site-packages (from stack-data->ipython>=4.0.0->ipywidgets) (0.8.3)
    Requirement already satisfied: asttokens in c:\users\kavya\anaconda3\lib\site-packages (from stack-data->ipython>=4.0.0->ipywidgets) (2.0.5)
    Requirement already satisfied: pure-eval in c:\users\kavya\anaconda3\lib\site-packages (from stack-data->ipython>=4.0.0->ipywidgets) (0.2.2)
    Requirement already satisfied: anyio>=3.1.0 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (4.2.0)
    Requirement already satisfied: argon2-cffi>=21.1 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (21.3.0)
    Requirement already satisfied: jinja2>=3.0.3 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (3.1.4)
    Requirement already satisfied: jupyter-client>=7.4.4 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (8.6.0)
    Requirement already satisfied: jupyter-core!=5.0.*,>=4.12 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (5.7.2)
    Requirement already satisfied: jupyter-events>=0.9.0 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (0.10.0)
    Requirement already satisfied: jupyter-server-terminals>=0.4.4 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (0.4.4)
    Requirement already satisfied: nbconvert>=6.4.4 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (7.16.4)
    Requirement already satisfied: nbformat>=5.3.0 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (5.10.4)
    Requirement already satisfied: overrides>=5.0 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (7.4.0)
    Requirement already satisfied: packaging>=22.0 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (24.1)
    Requirement already satisfied: prometheus-client>=0.9 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (0.14.1)
    Requirement already satisfied: pywinpty>=2.0.1 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (2.0.10)
    Requirement already satisfied: pyzmq>=24 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (25.1.2)
    Requirement already satisfied: send2trash>=1.8.2 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (1.8.2)
    Requirement already satisfied: terminado>=0.8.3 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (0.17.1)
    Requirement already satisfied: websocket-client>=1.7 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (1.8.0)
    Requirement already satisfied: async-lru>=1.0.0 in c:\users\kavya\anaconda3\lib\site-packages (from jupyterlab<4.3,>=4.2.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (2.0.4)
    Requirement already satisfied: httpx>=0.25.0 in c:\users\kavya\anaconda3\lib\site-packages (from jupyterlab<4.3,>=4.2.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (0.27.0)
    Requirement already satisfied: ipykernel>=6.5.0 in c:\users\kavya\anaconda3\lib\site-packages (from jupyterlab<4.3,>=4.2.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (6.28.0)
    Requirement already satisfied: jupyter-lsp>=2.0.0 in c:\users\kavya\anaconda3\lib\site-packages (from jupyterlab<4.3,>=4.2.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (2.2.0)
    Requirement already satisfied: setuptools>=40.1.0 in c:\users\kavya\anaconda3\lib\site-packages (from jupyterlab<4.3,>=4.2.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (75.1.0)
    Requirement already satisfied: babel>=2.10 in c:\users\kavya\anaconda3\lib\site-packages (from jupyterlab-server<3,>=2.27.1->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (2.11.0)
    Requirement already satisfied: json5>=0.9.0 in c:\users\kavya\anaconda3\lib\site-packages (from jupyterlab-server<3,>=2.27.1->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (0.9.6)
    Requirement already satisfied: jsonschema>=4.18.0 in c:\users\kavya\anaconda3\lib\site-packages (from jupyterlab-server<3,>=2.27.1->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (4.23.0)
    Requirement already satisfied: requests>=2.31 in c:\users\kavya\anaconda3\lib\site-packages (from jupyterlab-server<3,>=2.27.1->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (2.32.3)
    Requirement already satisfied: six in c:\users\kavya\anaconda3\lib\site-packages (from asttokens->stack-data->ipython>=4.0.0->ipywidgets) (1.16.0)
    Requirement already satisfied: idna>=2.8 in c:\users\kavya\anaconda3\lib\site-packages (from anyio>=3.1.0->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (3.7)
    Requirement already satisfied: sniffio>=1.1 in c:\users\kavya\anaconda3\lib\site-packages (from anyio>=3.1.0->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (1.3.0)
    Requirement already satisfied: argon2-cffi-bindings in c:\users\kavya\anaconda3\lib\site-packages (from argon2-cffi>=21.1->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (21.2.0)
    Requirement already satisfied: pytz>=2015.7 in c:\users\kavya\anaconda3\lib\site-packages (from babel>=2.10->jupyterlab-server<3,>=2.27.1->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (2024.1)
    Requirement already satisfied: certifi in c:\users\kavya\anaconda3\lib\site-packages (from httpx>=0.25.0->jupyterlab<4.3,>=4.2.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (2024.12.14)
    Requirement already satisfied: httpcore==1.* in c:\users\kavya\anaconda3\lib\site-packages (from httpx>=0.25.0->jupyterlab<4.3,>=4.2.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (1.0.2)
    Requirement already satisfied: h11<0.15,>=0.13 in c:\users\kavya\anaconda3\lib\site-packages (from httpcore==1.*->httpx>=0.25.0->jupyterlab<4.3,>=4.2.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (0.14.0)
    Requirement already satisfied: debugpy>=1.6.5 in c:\users\kavya\anaconda3\lib\site-packages (from ipykernel>=6.5.0->jupyterlab<4.3,>=4.2.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (1.6.7)
    Requirement already satisfied: nest-asyncio in c:\users\kavya\anaconda3\lib\site-packages (from ipykernel>=6.5.0->jupyterlab<4.3,>=4.2.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (1.6.0)
    Requirement already satisfied: psutil in c:\users\kavya\anaconda3\lib\site-packages (from ipykernel>=6.5.0->jupyterlab<4.3,>=4.2.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (5.9.0)
    Requirement already satisfied: MarkupSafe>=2.0 in c:\users\kavya\anaconda3\lib\site-packages (from jinja2>=3.0.3->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (2.1.3)
    Requirement already satisfied: attrs>=22.2.0 in c:\users\kavya\anaconda3\lib\site-packages (from jsonschema>=4.18.0->jupyterlab-server<3,>=2.27.1->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (23.1.0)
    Requirement already satisfied: jsonschema-specifications>=2023.03.6 in c:\users\kavya\anaconda3\lib\site-packages (from jsonschema>=4.18.0->jupyterlab-server<3,>=2.27.1->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (2023.7.1)
    Requirement already satisfied: referencing>=0.28.4 in c:\users\kavya\anaconda3\lib\site-packages (from jsonschema>=4.18.0->jupyterlab-server<3,>=2.27.1->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (0.30.2)
    Requirement already satisfied: rpds-py>=0.7.1 in c:\users\kavya\anaconda3\lib\site-packages (from jsonschema>=4.18.0->jupyterlab-server<3,>=2.27.1->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (0.10.6)
    Requirement already satisfied: python-dateutil>=2.8.2 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-client>=7.4.4->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (2.9.0.post0)
    Requirement already satisfied: platformdirs>=2.5 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-core!=5.0.*,>=4.12->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (3.10.0)
    Requirement already satisfied: pywin32>=300 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-core!=5.0.*,>=4.12->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (305.1)
    Requirement already satisfied: python-json-logger>=2.0.4 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-events>=0.9.0->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (2.0.7)
    Requirement already satisfied: pyyaml>=5.3 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-events>=0.9.0->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (6.0.1)
    Requirement already satisfied: rfc3339-validator in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-events>=0.9.0->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (0.1.4)
    Requirement already satisfied: rfc3986-validator>=0.1.1 in c:\users\kavya\anaconda3\lib\site-packages (from jupyter-events>=0.9.0->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (0.1.1)
    Requirement already satisfied: beautifulsoup4 in c:\users\kavya\anaconda3\lib\site-packages (from nbconvert>=6.4.4->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (4.12.3)
    Requirement already satisfied: bleach!=5.0.0 in c:\users\kavya\anaconda3\lib\site-packages (from nbconvert>=6.4.4->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (4.1.0)
    Requirement already satisfied: defusedxml in c:\users\kavya\anaconda3\lib\site-packages (from nbconvert>=6.4.4->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (0.7.1)
    Requirement already satisfied: jupyterlab-pygments in c:\users\kavya\anaconda3\lib\site-packages (from nbconvert>=6.4.4->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (0.1.2)
    Requirement already satisfied: mistune<4,>=2.0.3 in c:\users\kavya\anaconda3\lib\site-packages (from nbconvert>=6.4.4->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (2.0.4)
    Requirement already satisfied: nbclient>=0.5.0 in c:\users\kavya\anaconda3\lib\site-packages (from nbconvert>=6.4.4->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (0.8.0)
    Requirement already satisfied: pandocfilters>=1.4.1 in c:\users\kavya\anaconda3\lib\site-packages (from nbconvert>=6.4.4->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (1.5.0)
    Requirement already satisfied: tinycss2 in c:\users\kavya\anaconda3\lib\site-packages (from nbconvert>=6.4.4->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (1.2.1)
    Requirement already satisfied: fastjsonschema>=2.15 in c:\users\kavya\anaconda3\lib\site-packages (from nbformat>=5.3.0->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (2.16.2)
    Requirement already satisfied: charset-normalizer<4,>=2 in c:\users\kavya\anaconda3\lib\site-packages (from requests>=2.31->jupyterlab-server<3,>=2.27.1->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (3.3.2)
    Requirement already satisfied: urllib3<3,>=1.21.1 in c:\users\kavya\anaconda3\lib\site-packages (from requests>=2.31->jupyterlab-server<3,>=2.27.1->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (2.2.3)
    Requirement already satisfied: webencodings in c:\users\kavya\anaconda3\lib\site-packages (from bleach!=5.0.0->nbconvert>=6.4.4->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (0.5.1)
    Requirement already satisfied: fqdn in c:\users\kavya\anaconda3\lib\site-packages (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.9.0->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (1.5.1)
    Requirement already satisfied: isoduration in c:\users\kavya\anaconda3\lib\site-packages (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.9.0->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (20.11.0)
    Requirement already satisfied: jsonpointer>1.13 in c:\users\kavya\anaconda3\lib\site-packages (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.9.0->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (2.1)
    Requirement already satisfied: uri-template in c:\users\kavya\anaconda3\lib\site-packages (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.9.0->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (1.3.0)
    Requirement already satisfied: webcolors>=24.6.0 in c:\users\kavya\anaconda3\lib\site-packages (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.9.0->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (24.11.1)
    Requirement already satisfied: cffi>=1.0.1 in c:\users\kavya\anaconda3\lib\site-packages (from argon2-cffi-bindings->argon2-cffi>=21.1->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (1.17.1)
    Requirement already satisfied: soupsieve>1.2 in c:\users\kavya\anaconda3\lib\site-packages (from beautifulsoup4->nbconvert>=6.4.4->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (2.5)
    Requirement already satisfied: pycparser in c:\users\kavya\anaconda3\lib\site-packages (from cffi>=1.0.1->argon2-cffi-bindings->argon2-cffi>=21.1->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (2.21)
    Requirement already satisfied: arrow>=0.15.0 in c:\users\kavya\anaconda3\lib\site-packages (from isoduration->jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.9.0->jupyter-server<3,>=2.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.6->ipywidgets) (1.2.3)
    Note: you may need to restart the kernel to use updated packages.
    


```python
jupyter nbextension enable --py widgetsnbextension
jupyter nbextension install --py widgetsnbextension

```


      Cell In[31], line 1
        jupyter nbextension enable --py widgetsnbextension
                ^
    SyntaxError: invalid syntax
    



```python
import ipywidgets as widgets
from IPython.display import display

movie_input = widgets.Text(
    value="Toy Story",
    description="Movie Title:",
    disabled=False
)
movie_list = widgets.Output()

def on_type(data):
    with movie_list:
        movie_list.clear_output()
        title = data["new"]
        if len(title) > 5:
            display(search(title))

movie_input.observe(on_type, names='value')


display(movie_input, movie_list)
```


    Text(value='Toy Story', description='Movie Title:')



    Output()



```python
import ipywidgets as widgets
from IPython.display import display

# Dummy search function that returns a message
def search(title):
    return f"Searching for: {title}"

# Create a text input widget
movie_input = widgets.Text(
    value="Toy Story",
    description="Movie Title:",
    disabled=False
)
movie_list = widgets.Output()

# Event handler for text input
def on_type(data):
    with movie_list:
        movie_list.clear_output()  # Clear previous output
        title = data["new"]
        if len(title) > 5:  # Check if title length > 5
            result = search(title)
            display(result)  # Display the search result

movie_input.observe(on_type, names='value')

# Display the widgets
display(movie_input, movie_list)

```


    Text(value='Toy Story', description='Movie Title:')



    Output()



```python
import pandas as pd  # Importing pandas

# Now, read the CSV file
ratings = pd.read_csv("ratings.csv")



```

       userId  movieId  rating   timestamp
    0       1      296     5.0  1147880044
    1       1      306     3.5  1147868817
    2       1      307     5.0  1147868828
    3       1      665     5.0  1147878820
    4       1      899     3.5  1147868510
    


```python

ratings.dtypes
```




    userId         int64
    movieId        int64
    rating       float64
    timestamp      int64
    dtype: object




```python
similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()
```


    ----------------------------------------------------------------

    NameError                      Traceback (most recent call last)

    Cell In[7], line 1
    ----> 1 similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()
    

    NameError: name 'movie_id' is not defined



```python
movie_id=1
```


```python
similar_users=ratings[(ratings["movieId"]==movie_id) & (ratings["rating"]>4)]["userId"].unique()
```


```python
similar_users
```




    array([    36,     75,     86, ..., 162527, 162530, 162533], dtype=int64)




```python
similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]
```


```python
similar_user_recs
```




    5101            1
    5105           34
    5111          110
    5114          150
    5127          260
                ...  
    24998854    60069
    24998861    67997
    24998876    78499
    24998884    81591
    24998888    88129
    Name: movieId, Length: 1358326, dtype: int64




```python
similar_user_recs = similar_user_recs.value_counts() / len(similar_users)

similar_user_recs = similar_user_recs[similar_user_recs > .10]
```


```python
all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]
```


```python
all_users_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
```


```python
all_users_recs
```


    ----------------------------------------------------------------

    NameError                      Traceback (most recent call last)

    Cell In[23], line 1
    ----> 1 all_users_recs
    

    NameError: name 'all_users_recs' is not defined



```python
all_users_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
all_users_recs
```




    movieId
    318      0.342220
    296      0.284674
    2571     0.244033
    356      0.235266
    593      0.225909
               ...   
    551      0.040918
    50872    0.039111
    745      0.037031
    78499    0.035131
    2355     0.025091
    Name: count, Length: 113, dtype: float64




```python
rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
rec_percentages.columns = ["similar", "all"]
```


```python

rec_percentages
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>similar</th>
      <th>all</th>
      <th>score</th>
    </tr>
    <tr>
      <th>movieId</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1.000000</td>
      <td>0.124728</td>
      <td>8.017414</td>
    </tr>
    <tr>
      <th>3114</th>
      <td>0.280648</td>
      <td>0.053706</td>
      <td>5.225654</td>
    </tr>
    <tr>
      <th>2355</th>
      <td>0.110539</td>
      <td>0.025091</td>
      <td>4.405452</td>
    </tr>
    <tr>
      <th>78499</th>
      <td>0.152960</td>
      <td>0.035131</td>
      <td>4.354038</td>
    </tr>
    <tr>
      <th>4886</th>
      <td>0.235147</td>
      <td>0.070811</td>
      <td>3.320783</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2858</th>
      <td>0.216724</td>
      <td>0.167634</td>
      <td>1.292845</td>
    </tr>
    <tr>
      <th>296</th>
      <td>0.367295</td>
      <td>0.284674</td>
      <td>1.290232</td>
    </tr>
    <tr>
      <th>79132</th>
      <td>0.166817</td>
      <td>0.131384</td>
      <td>1.269693</td>
    </tr>
    <tr>
      <th>4973</th>
      <td>0.142501</td>
      <td>0.112405</td>
      <td>1.267747</td>
    </tr>
    <tr>
      <th>2959</th>
      <td>0.262649</td>
      <td>0.216717</td>
      <td>1.211946</td>
    </tr>
  </tbody>
</table>
<p>113 rows × 3 columns</p>
</div>




```python
rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
```


```python
rec_percentages = rec_percentages.sort_values("score", ascending=False)
```


```python
rec_percentages

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>similar</th>
      <th>all</th>
      <th>score</th>
    </tr>
    <tr>
      <th>movieId</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1.000000</td>
      <td>0.124728</td>
      <td>8.017414</td>
    </tr>
    <tr>
      <th>3114</th>
      <td>0.280648</td>
      <td>0.053706</td>
      <td>5.225654</td>
    </tr>
    <tr>
      <th>2355</th>
      <td>0.110539</td>
      <td>0.025091</td>
      <td>4.405452</td>
    </tr>
    <tr>
      <th>78499</th>
      <td>0.152960</td>
      <td>0.035131</td>
      <td>4.354038</td>
    </tr>
    <tr>
      <th>4886</th>
      <td>0.235147</td>
      <td>0.070811</td>
      <td>3.320783</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2858</th>
      <td>0.216724</td>
      <td>0.167634</td>
      <td>1.292845</td>
    </tr>
    <tr>
      <th>296</th>
      <td>0.367295</td>
      <td>0.284674</td>
      <td>1.290232</td>
    </tr>
    <tr>
      <th>79132</th>
      <td>0.166817</td>
      <td>0.131384</td>
      <td>1.269693</td>
    </tr>
    <tr>
      <th>4973</th>
      <td>0.142501</td>
      <td>0.112405</td>
      <td>1.267747</td>
    </tr>
    <tr>
      <th>2959</th>
      <td>0.262649</td>
      <td>0.216717</td>
      <td>1.211946</td>
    </tr>
  </tbody>
</table>
<p>113 rows × 3 columns</p>
</div>




```python
rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>similar</th>
      <th>all</th>
      <th>score</th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.000000</td>
      <td>0.124728</td>
      <td>8.017414</td>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>3021</th>
      <td>0.280648</td>
      <td>0.053706</td>
      <td>5.225654</td>
      <td>3114</td>
      <td>Toy Story 2 (1999)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>2264</th>
      <td>0.110539</td>
      <td>0.025091</td>
      <td>4.405452</td>
      <td>2355</td>
      <td>Bug's Life, A (1998)</td>
      <td>Adventure|Animation|Children|Comedy</td>
    </tr>
    <tr>
      <th>14813</th>
      <td>0.152960</td>
      <td>0.035131</td>
      <td>4.354038</td>
      <td>78499</td>
      <td>Toy Story 3 (2010)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy|IMAX</td>
    </tr>
    <tr>
      <th>4780</th>
      <td>0.235147</td>
      <td>0.070811</td>
      <td>3.320783</td>
      <td>4886</td>
      <td>Monsters, Inc. (2001)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>580</th>
      <td>0.216618</td>
      <td>0.067513</td>
      <td>3.208539</td>
      <td>588</td>
      <td>Aladdin (1992)</td>
      <td>Adventure|Animation|Children|Comedy|Musical</td>
    </tr>
    <tr>
      <th>6258</th>
      <td>0.228139</td>
      <td>0.072268</td>
      <td>3.156862</td>
      <td>6377</td>
      <td>Finding Nemo (2003)</td>
      <td>Adventure|Animation|Children|Comedy</td>
    </tr>
    <tr>
      <th>587</th>
      <td>0.179400</td>
      <td>0.059977</td>
      <td>2.991150</td>
      <td>595</td>
      <td>Beauty and the Beast (1991)</td>
      <td>Animation|Children|Fantasy|Musical|Romance|IMAX</td>
    </tr>
    <tr>
      <th>8246</th>
      <td>0.203504</td>
      <td>0.068453</td>
      <td>2.972889</td>
      <td>8961</td>
      <td>Incredibles, The (2004)</td>
      <td>Action|Adventure|Animation|Children|Comedy</td>
    </tr>
    <tr>
      <th>359</th>
      <td>0.253411</td>
      <td>0.085764</td>
      <td>2.954762</td>
      <td>364</td>
      <td>Lion King, The (1994)</td>
      <td>Adventure|Animation|Children|Drama|Musical|IMAX</td>
    </tr>
  </tbody>
</table>
</div>




```python
def find_similar_movies(movie_id):
    similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)

    similar_user_recs = similar_user_recs[similar_user_recs > .10]
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]
    
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending=False)
    return rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")[["score", "title", "genres"]]
```


```python

movie_name_input = widgets.Text(
    value='Toy Story',
    description='Movie Title:',
    disabled=False
)
recommendation_list = widgets.Output()

def on_type(data):
    with recommendation_list:
        recommendation_list.clear_output()
        title = data["new"]
        if len(title) > 5:
            results = search(title)
            movie_id = results.iloc[0]["movieId"]
            display(find_similar_movies(movie_id))

movie_name_input.observe(on_type, names='value')

display(movie_name_input, recommendation_list)
```


    Text(value='Toy Story', description='Movie Title:')



    Output()



```python
import ipywidgets as widgets
from IPython.display import display
import pandas as pd

# Load the movies dataset
movies = pd.read_csv("movies.csv")  # Ensure the path is correct
def search(title):
    title = title.strip().lower()
    movies["clean_title"] = movies["title"].str.lower().str.strip()
    results = movies[movies["clean_title"].str.contains(title, na=False)]
    return results


# Dummy function to display similar movies (replace with actual logic)
def find_similar_movies(movie_id):
    # For now, just display the movie ID
    return f"Similar movies for Movie ID: {movie_id}"

# Widgets for input and output
movie_name_input = widgets.Text(
    value='Toy Story',
    description='Movie Title:',
    disabled=False
)
recommendation_list = widgets.Output()

# Event handler for input changes
def on_type(data):
    with recommendation_list:
        recommendation_list.clear_output()
        title = data["new"]
        if len(title) > 5:
            results = search(title)
            if not results.empty:  # Check if results are not empty
                movie_id = results.iloc[0]["movieId"]
                display(find_similar_movies(movie_id))
            else:
                print("No results found.")

movie_name_input.observe(on_type, names='value')

display(movie_name_input, recommendation_list)

```


    Text(value='Toy Story', description='Movie Title:')



    Output()



```python
import ipywidgets as widgets
from IPython.display import display

movie_name_input = widgets.Text(
    value='Toy Story',
    description='Movie Title:',
    disabled=False
)
recommendation_list = widgets.Output()

def on_type(data):
    with recommendation_list:
        recommendation_list.clear_output()
        title = data["new"]
        if len(title) > 5:
            results = search(title)
            movie_id = results.iloc[0]["movieId"]
            display(find_similar_movies(movie_id))

movie_name_input.observe(on_type, names='value')

display(movie_name_input, recommendation_list)

```


    Text(value='Toy Story', description='Movie Title:')



    Output()



```python
movie_name_input = widgets.Text(
    value='Toy Story',
    description='Movie Title:',
    disabled=False
)
recommendation_list = widgets.Output()

def on_type(data):
    with recommendation_list:
        recommendation_list.clear_output()
        title = data["new"]
        if len(title) > 5:
            results = search(title)
            movie_id = results.iloc[0]["movieId"]
            display(find_similar_movies(movie_id))

movie_name_input.observe(on_type, names='value')

display(movie_name_input, recommendation_list)
```


    Text(value='Toy Story', description='Movie Title:')



    Output()



```python
import re

def search(title):
    title = title.strip().lower()
    # Escape special characters in the title
    escaped_title = re.escape(title)
    movies["clean_title"] = movies["title"].str.lower().str.strip()
    results = movies[movies["clean_title"].str.contains(escaped_title, na=False)]
    return results

```


```python
def on_type(data):
    with recommendation_list:
        recommendation_list.clear_output()
        title = data["new"]
        if len(title) > 5:
            results = search(title)
            if results.empty:
                print(f"No results found for '{title}'")
                return
            # Safely access the first result
            movie_id = results.iloc[0]["movieId"]
            display(find_similar_movies(movie_id))

```


```python
print(movies[movies["title"].str.contains("Avengers", case=False, na=False)])

```

           movieId                                              title  \
    2063      2153                               Avengers, The (1998)   
    17067    89745                               Avengers, The (2012)   
    22599   115727  Crippled Avengers (Can que) (Return of the 5 D...   
    25058   122892                     Avengers: Age of Ultron (2015)   
    25067   122912             Avengers: Infinity War - Part I (2018)   
    25068   122914            Avengers: Infinity War - Part II (2019)   
    30333   135979           Next Avengers: Heroes of Tomorrow (2008)   
    30431   136257                              Avengers Grimm (2015)   
    34536   145676                                  3 Avengers (1964)   
    35219   147238  The New Adventures of the Elusive Avengers (1968)   
    35372   147657                             Masked Avengers (1981)   
    40636   159920                            Shaolin Avengers (1994)   
    40637   159922                        The Shaolin Avengers (1976)   
    45097   169616                                  Scavengers (2013)   
    45394   170297                         Ultimate Avengers 2 (2006)   
    52858   186233                              The Scavengers (1970)   
    53341   187221  LEGO Marvel Super Heroes: Avengers Reassembled...   
    54283   189217                   Avengers Grimm: Time Wars (2018)   
    
                                     genres  \
    2063                   Action|Adventure   
    17067      Action|Adventure|Sci-Fi|IMAX   
    22599                  Action|Adventure   
    25058           Action|Adventure|Sci-Fi   
    25067           Action|Adventure|Sci-Fi   
    25068           Action|Adventure|Sci-Fi   
    30333  Action|Animation|Children|Sci-Fi   
    30431          Action|Adventure|Fantasy   
    34536                (no genres listed)   
    35219         Action|Adventure|Children   
    35372                            Action   
    40636                            Action   
    40637                            Action   
    45097                     Action|Sci-Fi   
    45394           Action|Animation|Sci-Fi   
    52858                             Drama   
    53341                         Animation   
    54283          Action|Adventure|Fantasy   
    
                                                 clean_title  
    2063                                avengers, the (1998)  
    17067                               avengers, the (2012)  
    22599  crippled avengers (can que) (return of the 5 d...  
    25058                     avengers: age of ultron (2015)  
    25067             avengers: infinity war - part i (2018)  
    25068            avengers: infinity war - part ii (2019)  
    30333           next avengers: heroes of tomorrow (2008)  
    30431                              avengers grimm (2015)  
    34536                                  3 avengers (1964)  
    35219  the new adventures of the elusive avengers (1968)  
    35372                             masked avengers (1981)  
    40636                            shaolin avengers (1994)  
    40637                        the shaolin avengers (1976)  
    45097                                  scavengers (2013)  
    45394                         ultimate avengers 2 (2006)  
    52858                              the scavengers (1970)  
    53341  lego marvel super heroes: avengers reassembled...  
    54283                   avengers grimm: time wars (2018)  
    


```python
print(movies["title"].unique())

```

    ['Toy Story (1995)' 'Jumanji (1995)' 'Grumpier Old Men (1995)' ...
     'Bad Poems (2018)' 'A Girl Thing (2001)' "Women of Devil's Island (1962)"]
    


```python
def search(title):
    title = title.strip().lower()
    # Use contains with case insensitive search
    movies["clean_title"] = movies["title"].str.lower().str.strip()
    results = movies[movies["clean_title"].str.contains(title, na=False, regex=False)]
    return results

```


```python
import re
import ipywidgets as widgets
from IPython.display import display, HTML


# Ensure movies DataFrame is loaded with necessary columns (movieId, title)
movies = pd.read_csv("movies_file.csv")

# Input Field for Movie Title
movie_name_input = widgets.Text(
    value='Toy Story',
    description='Movie Title:',
    disabled=False
)
recommendation_list = widgets.Output()

# Improved Search Function with Fuzzy Matching and Case Insensitivity
def search(title):
    title = title.strip().lower()
    movies["clean_title"] = movies["title"].str.lower().str.strip()
    # Calculate similarity score for fuzzy matching
    movies["similarity"] = movies["clean_title"].apply(lambda x: fuzz.partial_ratio(x, title))
    # Filter and sort by relevance
    results = movies[movies["similarity"] > 70].sort_values(by="similarity", ascending=False)
    return results

# Enhanced Event Handler with Error Handling and Clear Output
def on_type(data):
    with recommendation_list:
        recommendation_list.clear_output()
        title = data["new"]
        if len(title) > 2:  # Start searching after 2 characters for better UX
            display(HTML("<b>Searching...</b>"))  # Loading message
            results = search(title)
            recommendation_list.clear_output()
            
            # Check if results are found
            if not results.empty:
                try:
                    movie_id = results.iloc[0]["movieId"]
                    display(HTML(f"<h3>Similar Movies for '{title}':</h3>"))
                    display(find_similar_movies(movie_id))
                except IndexError:
                    display(HTML(f"<b>No results found for '{title}'</b>"))
            else:
                display(HTML(f"<b>No results found for '{title}'</b>"))

# Observe changes in the input field
movie_name_input.observe(on_type, names='value')

# Display the input and recommendation list
display(movie_name_input, recommendation_list)

```


    Text(value='Toy Story', description='Movie Title:')



    Output()



```python
import ipywidgets as widgets
from IPython.display import display, HTML
import pandas as pd

# Load your movie dataset (Ensure the file path is correct)
movies = pd.read_csv("movies.csv")

# Input Field for Movie Title
movie_name_input = widgets.Text(
    value='Toy Story',
    description='Movie Title:',
    disabled=False
)
recommendation_list = widgets.Output()

# Simple Search Function with Case Insensitivity
def search(title):
    title = title.strip().lower()
    movies["clean_title"] = movies["title"].str.lower().str.strip()
    # Filter for partial matches
    results = movies[movies["clean_title"].str.contains(title, na=False)]
    return results

# Dummy Similar Movie Finder (Update this as per your logic)
def find_similar_movies(movie_id):
    # For demonstration, we take the next 5 movies as "similar"
    similar_movies = movies[movies["movieId"] != movie_id].head(5)
    return similar_movies[["movieId", "title","score","genres"]]

# Event Handler for User Input
def on_type(data):
    with recommendation_list:
        recommendation_list.clear_output()
        title = data["new"]
        if len(title) > 2:  # Start searching after 2 characters for better UX
            display(HTML("<b>Searching...</b>"))  # Loading message
            results = search(title)
            recommendation_list.clear_output()
            
            # Check if results are found
            if not results.empty:
                try:
                    movie_id = results.iloc[0]["movieId"]
                    display(HTML(f"<h3>Similar Movies for '{title}':</h3>"))
                    similar_movies = find_similar_movies(movie_id)
                    
                    # Check if similar movies were found
                    if not similar_movies.empty:
                        display(similar_movies)
                    else:
                        display(HTML("<b>No similar movies found.</b>"))
                except IndexError:
                    display(HTML(f"<b>No results found for '{title}'</b>"))
            else:
                display(HTML(f"<b>No results found for '{title}'</b>"))

# Observe changes in the input field
movie_name_input.observe(on_type, names='value')

# Display the input and recommendation list
display(movie_name_input, recommendation_list)

```


    Text(value='Toy Story', description='Movie Title:')



    Output()



```python
import ipywidgets as widgets
from IPython.display import display, HTML
import pandas as pd

# Load your movie dataset (Ensure the file path is correct)
movies = pd.read_csv("movies.csv")
ratings= pd.read_csv("ratings.csv")

movie_scores = ratings.groupby('movieId')['rating'].mean().reset_index()
movie_scores.columns = ['movieId', 'score']

# Merge Scores with Movies DataFrame
movies = movies.merge(movie_scores, on='movieId', how='left')
# Input Field for Movie Title
movie_name_input = widgets.Text(
    value='Toy Story',
    description='Movie Title:',
    disabled=False
)
recommendation_list = widgets.Output()

# Simple Search Function with Case Insensitivity
def search(title):
    title = title.strip().lower()
    movies["clean_title"] = movies["title"].str.lower().str.strip()
    # Filter for partial matches
    results = movies[movies["clean_title"].str.contains(title, na=False)]
    return results

# Enhanced Similar Movie Finder
def find_similar_movies(movie_id):
    # For demonstration, returning next 5 movies as "similar"
    similar_movies = movies[movies["movieId"] != movie_id].head(5)
    
    # Display relevant columns (Removed 'score')
    return similar_movies[["movieId","title","score", "genres"]]

# Event Handler for User Input
def on_type(data):
    with recommendation_list:
        recommendation_list.clear_output()
        title = data["new"]
        if len(title) > 2:  # Start searching after 2 characters for better UX
            display(HTML("<b>Searching...</b>"))  # Loading message
            results = search(title)
            recommendation_list.clear_output()
            
            # Check if results are found
            if not results.empty:
                try:
                    movie_id = results.iloc[0]["movieId"]
                    display(HTML(f"<h3>Similar Movies for '{title}':</h3>"))
                    similar_movies = find_similar_movies(movie_id)
                    
                    # Check if similar movies were found
                    if not similar_movies.empty:
                        # Convert DataFrame to HTML table for display
                        display(HTML(similar_movies.to_html(index=False)))
                    else:
                        display(HTML("<b>No similar movies found.</b>"))
                except IndexError:
                    display(HTML(f"<b>No results found for '{title}'</b>"))
            else:
                display(HTML(f"<b>No results found for '{title}'</b>"))

# Observe changes in the input field
movie_name_input.observe(on_type, names='value')

# Display the input and recommendation list
display(movie_name_input, recommendation_list)

```


    Text(value='Toy Story', description='Movie Title:')



    Output()



```python
import ipywidgets as widgets
from IPython.display import display, HTML
import pandas as pd

# Load Movies and Ratings Datasets
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Calculate Score (Average Rating)
movie_scores = ratings.groupby('movieId')['rating'].mean().reset_index()
movie_scores.columns = ['movieId', 'score']

# Merge Scores with Movies DataFrame
movies = movies.merge(movie_scores, on='movieId', how='left')

# Input Field for Movie Title
movie_name_input = widgets.Text(
    value='Toy Story',
    description='Movie Title:',
    disabled=False
)
recommendation_list = widgets.Output()

# Simple Search Function with Case Insensitivity
def search(title):
    title = title.strip().lower()
    movies["clean_title"] = movies["title"].str.lower().str.strip()
    # Filter for partial matches
    results = movies[movies["clean_title"].str.contains(title, na=False)]
    return results

# Enhanced Similar Movie Finder
def find_similar_movies(movie_id):
    # For demonstration, returning next 5 movies as "similar"
    similar_movies = movies[movies["movieId"] != movie_id].head(5)
    
    # Display relevant columns including 'score'
    return similar_movies[["movieId","title", "genres", "score"]]

# Event Handler for User Input
def on_type(data):
    with recommendation_list:
        recommendation_list.clear_output()
        title = data["new"]
        if len(title) > 2:  # Start searching after 2 characters for better UX
            display(HTML("<b>Searching...</b>"))  # Loading message
            results = search(title)
            recommendation_list.clear_output()
            
            # Check if results are found
            if not results.empty:
                try:
                    movie_id = results.iloc[0]["movieId"]
                    display(HTML(f"<h3>Similar Movies for '{title}':</h3>"))
                    similar_movies = find_similar_movies(movie_id)
                    
                    # Check if similar movies were found
                    if not similar_movies.empty:
                        # Convert DataFrame to HTML table for display
                        display(HTML(similar_movies.to_html(index=False)))
                    else:
                        display(HTML("<b>No similar movies found.</b>"))
                except IndexError:
                    display(HTML(f"<b>No results found for '{title}'</b>"))
            else:
                display(HTML(f"<b>No results found for '{title}'</b>"))

# Observe changes in the input field
movie_name_input.observe(on_type, names='value')

# Display the input and recommendation list
display(movie_name_input, recommendation_list)

```


    Text(value='Toy Story', description='Movie Title:')



    Output()



```python

```
