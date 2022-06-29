# textPrep
a text preprocessing library for topic models

## textPrep for Norwegian subtitles
This fork of textPrep has extended the toolkit to work with subtitles from Norwegian TV programs. This includes extensions to existing rules, new rules, as well as some small improvements in code used for general purposes. Here is an exhaustive list of the changes:

### New rules
* **RemoveNumbers:** Removes all instances of numbers (digits) from text (`remove_numbers.py`)
* **RemoveSubtitleMetadata:** Removes metadata tags, characters and phrases that are found in Norwegian TV program subtitles (`remove_subtitle_metadata.py`)

### Extensions to existing rules
* **RemoveStopWords:** Extended to allow the use of a Norwegian stop word list for stop word removal (`remove_stopwords.py`)
* **Lemmatize:** Extended to allow lemmatization in other languages than English with the help of SpaCy language models (`lemmatize.py`)
* **PartOfSpeech:** Extended to allow PoS-tagging and removal in other languages than English with the help of SpaCy language models (`part_of_speech.py`)

### Other changes
* **get_data_stats:** This function adds support for calculating stop word statistics for datasets, using the extension above
* **word_co_frequency_document:** This function is added for calculating word co-frequencies in individual documents, to enable parallelizing (`common.py`)
* **compute_metrics:** Modified this function to return only coherence and diversity (`evaluate_topic_set.py`)

## Requirements
to install relevant requirements:
> pip install -r requirements.txt

Additional NLTK packages needed:
> stopwords
> 
> wordnet
> 
> averaged_perceptron_tagger

To install NLTK packages:
> python
```python
import nltk 
nltk.download()
```

Choose just the required packages (the whole set of additional NLTK data is massive)

### Using textPrep

#### Creating a pipeline and preprocessing
```python
from preprocessing_pipeline import (Preprocess, RemovePunctuation, Capitalization, RemoveStopWords, RemoveShortWords, TwitterCleaner, RemoveUrls)

# initialize the pipeline
pipeline = Preprocess()

# initialize the rules you want to use
rp = RemovePunctuation(keep_hashtags=False)
ru = RemoveUrls()
cap = Capitalization()

# include extra data in a rule if necessary
from nltk.corpus import stopwords
stopwords_list = stopwords.words('english')
stopwords_list.append(['rt', 'amp'])

rsw = RemoveStopWords(extra_sw=stopwords_list)

# add rules to the pipeline (the stringified rule makes it easy to save the pipeline details)
pipeline.document_methods = [(ru.remove_urls, str(ru),),
                             (rp.remove_punctuation, str(rp),),
                             (cap.lowercase, str(cap),),
                             (rsw.remove_stopwords, str(rsw),)
                             ]
```

You can load your data however you want, so long as it ends up as a list of lists. We provide methods for loading CSV files with and without dates.
```python
# load the data
def load_dataset_with_dates(path):
    dataset = []
    try:
        with open(path, 'r') as f:
            for line in f:
                dataset.append(line.strip().split('\t')[1].split(' '))
        return dataset
    except FileNotFoundError:
        print('The path provided for your dataset does not exist: {}'.format(path))
        import sys
        sys.exit()

dataset = load_dataset_with_dates('data/sample_tweets.csv')
# dataset[i] = ['list', 'of', 'words', 'in', 'document_i']

# initialize the pipeline runner
from preprocessing_pipeline.NextGen import NextGen

runner = NextGen()

# preprocess the data, with some extra ngrams thrown in to ensure they are considered regardless of frequency
processed_dataset = runner.full_preprocess(dataset, pipeline, ngram_min_freq=10, extra_bigrams=None, extra_ngrams=['donald$trump', 'joe$biden', 'new$york$city'])

# assess data quality quickly and easily
from evaluation_metrics.dataset_stats import get_data_stats
print(get_data_stats(processed_dataset))
```

You can do some extra filtering after preprocessing, like TF-IDF filtering
```python
from settings.common import word_tf_df

freq = {}
freq = word_tf_df(freq, processed_dataset)
filtered_dataset = runner.filter_by_tfidf(dataset=processed_dataset, freq=freq, threshold=0.25)

# assess data quality again 
from evaluation_metrics.dataset_stats import get_data_stats
print(get_data_stats(filtered_dataset))
```

### Referencing textPrep

```
Churchill, Rob and Singh, Lisa. 2021. textPrep: A Text Preprocessing Toolkit for Topic Modeling on Social Media Data. DATA 2021.
```

```bibtex 
@inproceedings{churchill2021textprep,
author = {Churchill, Rob and Singh, Lisa},
title = {textPrep: A Text Preprocessing Toolkit for Topic Modeling on Social Media Data},
booktitle = {DATA 2021},
year = {2021},
}
```
