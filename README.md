# observatory
Python package for collecting and analyzing webpages

## Modules
### `start_project`
Initializes a project directory

### `search_twitter`
Searches Twitter for tweets matching parameters including terms. By default, returns a pandas DataFrame of urls mentioned in tweets and timestamps of tweets. Twitter search credentials required.

### `twitter_process`
Compiles two or more CSVs of results from searching Twitter.

### `search_google`
Searches Google for terms. Google Custom Search Engine credentials required.

### `google_process`
Compiles results from multiple Google searches.

### `get_domains`
Extracts domain-level information from the urls returned by Google and/or Twitter searches (e.g. 'google' in www.google.com)

### `initialize_crawl`
Initializes a Scrapy crawl on a set of domains. Returns a JSON file of urls found through the crawl.

### `crawl_process`
Processes the JSON output of a crawl into a pandas DataFrame.

### `crawl`
Not implemented yet. 
`!scrapy crawl digcon_crawler -O output.json --nolog

### `search_merge`
Merges Google / Twitter searches and crawl results.

### `get_versions`
Gets historical versions of Twitter-searched urls using the Internet Archive's Wayback Machine. Attempts to find the version of the page archived closest in time to when it was tweeted.

### `initialize_scrape`
Initializes files to scrape urls for their HTML.

### `scrape`
Conducts the scrape of pages' HTML. Stores body text in a Postgresql database. 

### `query`
A set of methods for searching the Postgreql database of site text, including filtering empty results and counting specified search terms.

### `ground_truth`
Produces a sample of pages for verifying counts of terms.

### `analyze_twitter`
Calculates and visualizes averages and frequencies per year for each search term in the site text (Twitter-found sites only, since those are the only ones with timestamps)

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
