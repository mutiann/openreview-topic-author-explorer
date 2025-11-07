# OpenReview Topic Authors Explorer

Brief tools to:

1. `fetch.py`: fetch accepted/published papers from major ML venues hosted on OpenReview;
2. `export.py`: build a static webpage listing authors who have published papers on certain topics, according to provided search terms.

An example is provided in [here](authors_topic_linear_attention.html) for authors with papers on linear attention/state-space models.

# Usage

Please refer to the arguments in each script for the usage. By default, running `fetch.py` will
fetch papers on ICLR, ICML, NeurIPS, COLM, and TMLR for the past 3 years, and save it to `openreview_output`.
We've also pregenerated these files in November 2025, included in this repository for convenience.

Then, running `export.py` with `--terms` will generate a static HTML page listing authors who have published papers 
matching the specified terms in title/abstract/keywords. 

An example output file is `authors_topic_linear_attention.html`.
This is generated with
```python export.py --input openreview_output --terms "linear attention,mamba,state space model" --out authors_topic_linear_attention.html```

Dependencies are listed in `requirements.txt`.
Some OpenReview profiles can be accessed only after logging in. To enable this feature, please set the `OPENREVIEW_TOKEN`
environment variable with your OpenReview token before running `fetch.py`. The token can be found by running 
```
import openreview
client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')
print(client.login_user(username, password)['token'])
```