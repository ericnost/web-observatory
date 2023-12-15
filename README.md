# observatory
Python package for collecting and analyzing webpages

See [here](https://github.com/ericnost/digital_conservation) for extended examples of `observatory` in use.

## Modules
### `start_project`
Initializes a project directory

### `search_google`
Searches Google for terms. Google Custom Search Engine credentials required.

### `google_process`
Compiles results from multiple Google searches.

### `get_domains`
Extracts domain-level information from the urls returned by Google searches (e.g. 'google' in www.google.com)

### `initialize_crawl`
Initializes a Scrapy crawl on a set of domains. Returns a JSON file of urls found through the crawl.

### `crawl_process`
Processes the JSON output of a crawl into a pandas DataFrame.

### `crawl`
Not implemented yet. 
`!scrapy crawl digcon_crawler -O output.json --nolog

### `search_merge`
Merges Google searches and crawl results.

### `get_versions`
~Gets historical versions of Twitter-searched urls using the Internet Archive's Wayback Machine. Attempts to find the version of the page archived closest in time to when it was tweeted.~ \
Uses the `requests` package to ping the url and get the "full" address rather than a redirect (e.g. bit.ly/12312). This helps in scraping.

### `initialize_scrape`
Initializes files to scrape urls for their HTML.

### `scrape`
Conducts the scrape of pages' HTML. Stores body text in a Postgresql database. 

### `query`
A set of methods for searching the Postgreql database of site text, including filtering empty results and counting specified search terms.

### `ground_truth`
Produces a sample of pages for verifying counts of terms.

### `analyze_orgs`
Calculates and visualizes averages and frequencies for each search term in the site text and summarizes by organization (domain).

### `analzye_term_correlations`
Calculates and visualizes co-variance metrics for specified search terms in the site text. 

### `co_occurrence`
Returns specific pages using two or more specified search terms.

## TBD
- Documentation :(
- Convert modules to methods of data classes
- Add crawl module
