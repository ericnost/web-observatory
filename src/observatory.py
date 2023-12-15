# -*- coding: utf-8 -*-
# OBSERVATORY
"""
Documentation....
"""

# Set project name
project = "default"

# Set passwords
credentials = {"google":{"devkey": "", "cx": ""}, "postgres":{"user":"", "db":"", "password":""}}

def start_project(project):
  # Create project directory
  import os
  ## Parent Directory path
  cwd = os.getcwd()
  ## Path
  path = os.path.join(cwd, project)
  ## Create the directory if it doesn't exist
  if os.path.exists(path) == False:
    os.mkdir(path)
    print("Directory '% s' created" % path)
  print("Done!")

# Global imports
import pandas


"""## Google search (Python) v.1

### Set up Google search
"""

# Google search 
# From https://github.com/googleapis/google-api-python-client/blob/master/samples/customsearch/main.py
# See also https://stackoverflow.com/questions/41032472/how-to-query-an-advanced-search-with-google-customsearch-api

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2014 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Simple command-line example for Custom Search.
Command-line application that does a search.
"""

def search_google(q, project):
  __author__ = 'jcgregorio@google.com (Joe Gregorio)'

  import pprint
  #pip install google-api-python-client &>/dev/null;
  from googleapiclient.discovery import build
  import pandas

  results = []

  def get_google(q, n = 0):
    # Build a service object for interacting with the API. Visit
    # the Google APIs Console <http://code.google.com/apis/console>
    # to get an API key for your own application.
    service = build("customsearch", "v1",
              developerKey=credentials["google"]["devkey"])

    # Build and execute the query. For parameters, see: https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list
    res = service.cse().list(
        q = q, # Basic search term
        cx = credentials["google"]["cx"],
        #exactTerms = q, # Can use exact terms
        start = n,
      ).execute()
    #pprint.pprint(res) # Debugging
    
    for item in res["items"]: # For each result
      hit = {"date": None, "link": item["link"], "metrics": None, "source": "Google", "query": q} # Construct a dictionary of relevant results
      results.append(hit) # Add this item to all results
      #https://www.googleapis.com/customsearch/v1?q={searchTerms}&num={count?}&cx={cx?}&start={startIndex?}&alt=json
    
    # Set the start for the next query
    n+=10 
    print(n, end='\r')

    if n < 50: # Can go up to 100
      get_google(q, n) 
    else:
      print('Done')
      # Turn results into dataframe
      g = pandas.DataFrame.from_dict(results)
      #display(g) # Debugging
      g.to_csv(project+"/google_search_results_"+q+".csv")

  get_google(q)

"""### Process the Google search data"""

def google_process(filespath = None, datatype = "CSV", project = None):
  # compile CSVs from Python queries, JSONs from HTML/JS queries

  import glob
  import json
  import pandas

  if filespath == None:
    filespath = project+"/*google*.csv"
  
  filenames = glob.glob(filespath) #glob.glob("/content/*.json") #HMTL/JS only

  if datatype == "CSV":
    if len(filenames) > 1: # if we have multiple files
      combined_csv = pandas.concat( [ pandas.read_csv(f, header = 0) for f in filenames ] )
    else: # if we have just one
      combined_csv = pandas.read_csv(filenames[0], header=0)
    #combined_csv.columns = ['original_index', 'date', 'link', 'tweet_id', 'metrics', 'source', 'query'] # Rename columns (?) Do these positions change each query?!
    google = combined_csv
  
  elif datatype == "JSON":
    all_results = pandas.DataFrame()
    for f in filenames:
      these_results = json.load(open(f))
      try:
        q = these_results["queries"]["request"][0]["exactTerms"] 
        #print(q) # Debugging
      except KeyError:
        q = these_results["queries"]["request"][0]["searchTerms"] 
        #print(q) # Debugging
      try:
        these_results = pandas.DataFrame(these_results["items"])
        these_results["query"] = q
        all_results = pandas.concat([all_results, these_results])
      except KeyError:
        print("no items")
    google = all_results

  google = google.drop_duplicates(subset='link', keep="first") # keep first link out of duplicates
  google.reset_index(inplace=True, drop=True)
  google = google[['date','link','metrics','source','query']] # Drop unnecessary columns
  google.to_csv(project+"/google_search_results_compiled.csv") # Export compiled queries
  
  return google

"""
## Get domains ##

"""
def get_domains(pages):
  #from urllib.parse import urlparse
  import tldextract

  def domain(row):
    result = tldextract.extract(row['link'])
    domain = (result.domain)
    #parsed_uri = urlparse(row['link'])
    #domain = '{uri.scheme}://{uri.netloc}/'.format(uri=parsed_uri)
    #domain = domain.strip("https://")
    return domain
  
  pages["domain"] = pages.apply(lambda row: domain(row), axis=1)
  return pages


"""## CTRL-F v.2

### Crawling
"""
def initialize_crawl(sites, domains, project):

  # Create a new project
  ## Test if this has already been done
  import os
  if os.path.exists("digcon_crawler/digcon_crawler") == False: # Change to general name
    import subprocess
    subprocess.call(["scrapy", "startproject", "digcon_crawler"]) # Change digcon_crawler to project name?
  
  # Crawling spider - this one is useful for getting new pages
  crawler = """
  from scrapy.linkextractors import LinkExtractor
  from scrapy.spiders import Rule, CrawlSpider
  from scrapy import Request
  import logging
  from scrapy.utils.log import configure_logging 
  from digcon_crawler.items import DigconItem
  
  class DigconSpider(CrawlSpider):
    name = 'digcon_crawler'
    allowed_domains = """ + str(domains) + """
    start_urls = """ + str(sites) + """

    configure_logging(install_root_handler=False)
    logging.basicConfig(
      filename='log.txt',
      format='%(levelname)s: %(message)s',
      level=logging.WARNING
    )

    # This spider has one rule: extract all (unique and canonicalized) links, follow them and parse them using the parse_items method
    rules = [
      Rule(
        LinkExtractor(
          canonicalize=False,
          unique=True
        ),
        follow=True,
        callback="parse_items"
      )
    ]
  
    # Method which starts the requests by visiting all URLs specified in start_urls
    def start_requests(self):
      for url in self.start_urls:
        yield Request(url, dont_filter=True)

    # Method for parsing items
    def parse_items(self, response):
      # The list of items that are found on the particular page
      items = []
      # Only extract canonicalized and unique links (with respect to the current page)
      links = LinkExtractor(canonicalize=False, unique=True).extract_links(response)
      # Now go through all the found links
      for link in links:
        # Check whether the domain of the URL of the link is allowed; so whether it is in one of the allowed domains
        is_allowed = False
        for allowed_domain in self.allowed_domains:
          if allowed_domain in link.url:
            is_allowed = True
        # If it is allowed, create a new item and add it to the list of found items
        if is_allowed:
          item = DigconItem()
          item['url_from'] = response.url
          item['url_to'] = link.url
          items.append(item)
      # Return all the found items
      #print(items) # Debugging
      return items
  """
  with open('digcon_crawler/digcon_crawler/spiders/digcon_crawler.py', mode='w') as file: #TBD: generalize this
    file.write(crawler)

  items = """
  from scrapy.item import Item, Field
  
  class DigconItem(Item): #TBD: generalize this
    # The source URL
    url_from = Field()
    # The destination URL
    url_to = Field()
  """
  with open('digcon_crawler/digcon_crawler/items.py', mode='w') as file: #TBD: generalize this
    file.write(items)

  pipelines = """
  from scrapy.exceptions import DropItem
  
  class DuplicatesPipeline(object):
    def __init__(self):
      self.ids_seen = set()
  
    def process_item(self, item, spider):
      if item['url_from'] in self.ids_seen:
        raise DropItem("Duplicate item found: %s" % item)
      else:
        self.ids_seen.add(item['url_from'])
        return item
  """
  with open('digcon_crawler/digcon_crawler/pipelines.py', mode='w') as file:
    file.write(pipelines)

  # Upload settings and middlewares (from Github?)
  # Currently configured in Downloads folder....
  # Can copy/paste into folder when running locally

  print("Don't forget to move settings and middlewares")


"""### Processing Crawl Results"""

def crawl_process(file):
  #from urllib.parse import urlparse
  #https://pypi.org/project/tldextract/
  import tldextract 

  df = pandas.read_json(file, orient='records')
  crawl = pandas.DataFrame(set(df['url_from']).union(set(df['url_to']))).rename(columns={0: "link"})
  
  # Need to account for and track organizations!
  def domain(row):
    result = tldextract.extract(row['link'])
    domain = (result.domain)
    return domain
  crawl["query"] = crawl.apply(lambda row: domain(row), axis=1)
  crawl["source"] = "Crawl"

  return crawl


"""### Processing Searches and Crawls Together"""

# Merge results from Twitter and Google searches and Crawls
def search_merge(twitter = None, google = None, crawl = None, project = None):
  t, g, s = (None,)*3

  try:
    if twitter is not None:
      t = twitter.copy() # Create a copy of Twitter results
    if google is not None:
      g = google.copy() # Create a copy of Google results
    if crawl is not None:
      s = crawl.copy() # Create a copy of crawl results
    combo = [data for data in [t,g,s] if data is not None]
    result = pandas.concat(combo) # Combine
    result = result.drop_duplicates(subset=['link']) # De-duplicate here
    result = result.reset_index(drop=True) # Re-number the rows
    
    result.to_csv(project + "/all_search_results.csv")

    return result
  
  except:
    print("Did you enter data?")

"""### Get Historical Versions of Twitter Pages"""
    
def get_versions(pages, project = None):
  """ Get historical versions of Twitter urls before scraping"""

  # Exceptions to handle
  class Error(Exception):
    """Base class for other exceptions"""
    pass
  class NoContemporaryVersions(Error):
    """Raised when the WM has no snapshots for the given time period and site"""
    #print("No recent versions", end='\r') # Debugging
    pass
  class NoVersions(Error):
    """Raised when the WM has no snapshots at all"""
    #print("No versions", end='\r') # Debugging
    pass

  # Requirements
  import time
  from datetime import date, timedelta, datetime
  import requests
  import fnmatch
  from urllib.parse import urljoin, urlparse
  from wayback import WaybackClient

  # Set up
  pages["full_url"] = None # The actual url and not some bit.ly url 
  pages["url_meta"] = None # Possible additional information
  pages["wm_url"] = None # We may have this when we have previously saved pages to the WM
  pages["wm_meta"] = None # For storing information about the WM snapshot that was available
  
  # Helper function to try to get versions of the page from the Wayback Machine
  def wm_versions(index, versions):
    for n, version in enumerate(reversed(versions)): # For each version in all the snapshots, starting from the most recent
      if version.status_code == 200 or version.status_code == 301 or version.status_code == '-': 
        # If the IA snapshot was viable...200 = normal, 301 = redirect (many shortened bit.ly links will have this)
        url = version.raw_url
        return url
        break
      else:
        if n+1 == len(versions):
          # If we've gone through all the snapshots and there's not one we can get...
          pages.at[index,"wm_meta"] = "No Decodable Snapshot"
          #print("\tThere's no snapshot we can decode for", link, end='\r') # Debugging
          break      
        else:
          pass # Continue going through the versions

  # Loop through the pages
  for index, page in pages.iterrows():
    print(index, end='\r')
    
    # Backup every 5000 links...
    if index % 5000 == 0:
      pages.to_csv(project+"/page_versions_dump_"+str(index)+".csv") #Colab /content
      #print("Exported", end='\r') # Debugging

    #print(pages.iloc[index,[pages.columns.get_loc("link")]]) # Debugging
    #time.sleep(3) # Slow down requests
    
    # Get real link from ow.ly, bit.ly, etc. - this will help with WM lookups - and count (to know what we should archive for coding) 
    link = page["link"] # The link we'll start with is the link in this column
    try:
      response = requests.get(link, timeout=60) # Give it up to 60 seconds to get the link
      full_url = urljoin(response.request.url, urlparse(response.request.url).path) # Delete any query parameters
      pages.at[index,"full_url"] = full_url # Save url
    except:
      full_url = link
      pages.at[index,"url_meta"] = "Couldn't get full url"
      #print("Couldn't get full url", end='\r') # Debugging

    # Test if we have Twitter (historical) or other sources
    if page['source'] != "Twitter":
      # For links from Google or a crawl, we can do a simple request of that page
      # This could be modified or deleted so that all pages go through the Wayback Machine
      # This would be facilitated if they are saved there via ArchiveNow (see above module)
      pages.at[index,"wm_meta"] = "Not historical"

    # Otherwise, for Twitter sources, we'll use the Wayback Machine
    else:
      # Time parameters
      start = datetime.strptime(page["date"], "%Y-%m-%dT%H:%M:%S.%fZ") # Tweet date format
      end = start - timedelta(days=180) # Try past six months prior to tweet

      try:

        with WaybackClient() as client: # Use EDGI's Wayback Machine client to retrieve snapshots

          try: 
            dump = client.search(full_url, from_date = end, to_date = start) # To do: give it a timeout?        
            versions = list(dump)
            if len(versions) == 0: # No versions from six months prior to tweet
              raise NoContemporaryVersions
            else:
              try:
                #print("Time-of-Tweet Version", end='\r') # Debugging
                wm_url = wm_versions(index, versions) # Get a version that works # CHANGE TO VERSIONS_TESTER
                pages.at[index,"wm_url"] = wm_url
                pages.at[index,"wm_meta"] = "Time-of-Tweet Version"

              except: # There are current snapshots but they're not working; try older versions
                #print("Current snapshots but they're not working; try older versions", end='\r')
                raise NoContemporaryVersions

          except NoContemporaryVersions: # If there is no snapshot in six months prior to tweet, try all dates including now
            try:
              dump = client.search(full_url) # Try any snapshot. # To do: give it a timeout?
              versions = list(dump)
              if len(versions) == 0: # No versions at all
                raise NoVersions
              else:
                try:
                  #print("No contemporary versions", end='\r') # Debugging
                  wm_url = wm_versions(index, versions)
                  pages.at[index,"wm_url"] = wm_url
                  pages.at[index,"wm_meta"] = "No Time-of-Tweet Versions; Got Older Version"

                except: # If there are snapshots, but we're having trouble with them, try the live page
                  try:
                    #print("No working versions", end='\r') # Debugging
                    pages.at[index,"wm_meta"] = "No Working Wayback Version; Got Live URL"

                  except:
                    #print("Can't count terms at all :(", full_url, end='\r') # Debugging
                    pass

            except NoVersions: # If no snapshot at all, count current version of the page
              try:
                #print("No versions at all", end='\r') # Debugging
                pages.at[index,"wm_meta"] = "No Wayback Version; Got Live URL"
              except:
                #print("Can't count terms at all :(", full_url, end='\r') # Debugging
                pass

      except:
        #print("Other exception", end='\r') # Debugging
        pages.at[index,"wm_meta"] = "Exception"
        pages.to_csv(project+"/page_versions_dump_"+str(index)+".csv") # Save the data when exceptions occur
                  
  pages.to_csv(project+"/page_versions_full.csv")
  return pages


"""### Scraping"""
def initialize_scrape(versions = None, project = None):
  # Set up Scrapy to help crawl and scrape websites
  ## Test if this has already been done
  import os
  if os.path.exists("digcon_scraper/digcon_scraper") == False: #TBD: generalize this
    import subprocess
    subprocess.call(["scrapy", "startproject", "digcon_scraper"]) #TBD: generalize this

  # Commented out IPython magic to ensure Python compatibility.
  # Create database for results

  # Prepare data by extracting unique urls and domains
  import json
  import pandas
  import tldextract

  ## Create a new "scrape_url" column that is either the full_url or wm_url depending on what's available
  versions.loc[versions['wm_url'].isna(), 'scrape_url'] = versions["full_url"].str.lower() # Standardize by lower-casing
  versions.loc[~versions['wm_url'].isna(), 'scrape_url'] = versions["wm_url"].str.lower() # Standardize by lower-casing

  ## Then get domains
  def domain(row):
    if "web.archive.org" in str(row['scrape_url']):
      s = row['scrape_url'][45:] # Account for Wayback Machine
    else:
      s = row['scrape_url']
    try:
      result = tldextract.extract(s)
      domain = (result.domain)
    except:
      domain = "None"
    return domain
  versions['domain'] = versions.apply(lambda row: domain(row), axis=1)

  ## Filter and format
  ### Get a unique set of week / urls. This is to avoid over scraping/counting on the Twitter data.
  ### We will scrape the same link only once if it was shared in different weeks. We avoid scraping links shared back to back.
  ### These are only caught through the "full url" look up in get_versions. They often have different links e.g. bit.ly/1234 vs bit.ly/1531
  ### But have the same full_url e.g. microsoft.com. If they are in different weeks, we'll scrape them twice even if the wm_url is the same.
  versions['date'] = pandas.to_datetime(versions['date']).dt.strftime('%Y-%U') # Convert to year/week
  urls = versions.drop_duplicates(subset=['date', 'full_url'], keep='last') 
  urls = urls[['date', 'source', 'scrape_url', 'domain']]
  urls = urls.loc[~urls['scrape_url'].isna()] # Remove pages w/o a scrape
  urls = urls.to_json(orient="records")
  urls = json.loads(urls)

  items = """
# #/content/digcon/digcon/items.py # Colab
# # Save URL and body text
from scrapy.item import Item, Field
 
class DigconscrapeItem(Item):
   url = Field() # The source URL
  """
  with open('digcon_scraper/digcon_scraper/items.py', mode='w') as file: #TBD: generalize this
    file.write(items)

  print("Done!")
  return urls


def scrape(urls):
  # Scraping spider - this one is useful for visiting a set list of pages and scraping them
  #from twisted.internet import reactor
  import scrapy 
  import logging
  from scrapy.utils.log import configure_logging 
  from scrapy.crawler import CrawlerProcess #Runner

  import importlib.util
  import sys
  spec = importlib.util.spec_from_file_location("module.items", "digcon_scraper/digcon_scraper/items.py") # TBD: generalize this
  items = importlib.util.module_from_spec(spec)
  sys.modules["module.items"] = items
  spec.loader.exec_module(items)
  #from items import DigconscrapeItem # Modularization: change "digcon" throughout to "observatory" or let users name a project

  import re # for processing text

  import psycopg2

  conn = None 
  conn = psycopg2.connect("dbname="+credentials["postgres"]["db"]+" user="+credentials["postgres"]["user"]+" password="+credentials["postgres"]["password"]+" host=localhost port=5432")
  conn.autocommit = True

  cur = conn.cursor()
  cur.execute("""CREATE TABLE IF NOT EXISTS public.site_text (url varchar,  body_text varchar, date varchar, source varchar(10), domain varchar);""")

  def DBinsert(url, contents, date, source, domain, count):
    print("inserting: " + str(count), end='\r')
    sql = "INSERT INTO site_text(url,  body_text, date, source, domain) VALUES(%s,%s,%s,%s,%s);"
    try:
      cur.execute(sql, [url, contents, date, source, domain])
      conn.commit()
      #print("inserted", end='\r')
    except (Exception, psycopg2.DatabaseError) as error:
      print(error)
          
  class DigconscrapeSpider(scrapy.Spider): #TBD: generalize this
    name = 'digcon_scraper'
    
    def __init__(self, urls,  **kwargs): #domains = None,
      self.start_urls = urls
      super().__init__(**kwargs)  # https://stackoverflow.com/questions/15611605/how-to-pass-a-user-defined-argument-in-scrapy-spider
          
    configure_logging(install_root_handler=False)
    logging.basicConfig(
        filename='log.txt',
        format='%(levelname)s: %(message)s',
        level=logging.WARNING
    )
    
    def start_requests(self):
      count = 0
      for each in self.start_urls:
        count += 1
        each["count"] = count
        yield scrapy.Request(url = each['scrape_url'], meta = each, callback = self.parse_result)
          
    def parse_result(self, response):
      item = items.DigconscrapeItem()
      item['url'] = response.url
      
      try:
        # Work on the body of the page
        body = ' '.join(response.xpath("//body//p//text()").extract()) # without selenium
        body = body.lower() # Uncapitalize all the words for matching purposes
        body = re.sub(r'[^\w\s]','',body) # removes punctuation
        body = " ".join(line.strip() for line in body.split("\n")) # remove returns, adds spaces
        body = " ".join(body.split()) # removes extra spaces
        #item['body'] = body # Add to export/log instead of db
        # add total word count here?
        DBinsert(response.url, body, response.meta['date'], response.meta['source'], response.meta['domain'], response.meta['count']) # Add to DB
      except:
        pass

      return item
    
  process = CrawlerProcess({'FEED_FORMAT': 'json','FEED_URI': 'digcon_scrape.json', 'LOG_LEVEL': 'WARNING'}) # update feeds?
  process.crawl(DigconscrapeSpider, urls=urls)
  process.start()
  conn.close() # close connection

  print("Done!", end='\r')


"""### Querying Results in a Database"""

def query(qtype, terms = None, project = None):
  """
  qtypes: 
    copy = copy the full data to a separate table
    filter_empty = filter (a copy) to remove pages without text
    filter_words = filter (a copy) to remove (Twitter) pages without text
    get_words = return Pandas dataframe with word counts
  """

  # Connect to db
  import psycopg2
  conn = None 
  conn = psycopg2.connect("dbname="+credentials["postgres"]["db"]+" user="+credentials["postgres"]["user"]+" password="+credentials["postgres"]["password"]+" host=localhost port=5432")
  cur = conn.cursor()

  if qtype == "copy":
    sql = 'CREATE TABLE IF NOT EXISTS site_text_copy AS SELECT * FROM site_text';
    cur.execute(sql)
    conn.commit()
    cur.close()
    conn.close()
  
  elif qtype == "filter_empty":
    sql = 'DELETE FROM site_text_copy WHERE (body_text = \'\') IS TRUE;'
    cur.execute(sql)
    conn.commit()
    cur.close()
    conn.close()

  elif qtype == "filter_words":
    # Filter table by core terms - only keep _Twitter_ search pages that match.
    ## Split terms into something sql can manage
    ws = "'"
    for term in terms:
      try:
        ws += "" + term + "|"
      except:
        print("are your terms actually strings?")
    ws = ws[0:-1] + "'" # Remove last | 

    sql = 'DELETE FROM site_text_copy WHERE ((body_text ~* '+ws+') IS FALSE) AND (source = \'Twitter\')'
    cur.execute(sql)
    conn.commit()
    cur.close()
    conn.close()

  elif qtype == "get_words":
    # Get single word terms per page
    ## Split terms into something sql can manage
    #ws = "("
    #for term in terms:
    #  try:
    #    ws += "'" + term + "',"
    #  except:
    #    print("are your terms actually strings?")
    #ws = ws[0:-1] + ")"

    #sql = 'SELECT url, word, count(*), source, domain, date FROM (SELECT regexp_split_to_table(body_text, \'\s\') as word, url as url, source as source, domain as domain, date as date FROM site_text_copy) t WHERE word in '+ ws +' GROUP BY url, word, source, domain, date ORDER BY url, word'
    
    # Create columns and count terms
    for term in terms:
      # Columns
      sql = "ALTER TABLE site_text_copy ADD COLUMN IF NOT EXISTS \""+term+"\" int;"
      cur.execute(sql)
      conn.commit()
      #cur.close()
      
      # Count
      sql = """
      UPDATE site_text_copy
      SET \""""+term+"""\" = 
        (CHAR_LENGTH(body_text) - CHAR_LENGTH(REPLACE(body_text, \'"""+term+"""\', ''))) 
        / CHAR_LENGTH(\'"""+term+"""\');
      """ 
      #print(sql) # Debugging
      cur.execute(sql)
      conn.commit()
      #cur.close()

    sql = 'select "url", "domain", "source", "date", \"' + '\", \"'.join(terms) + '\" from site_text_copy'
    words = pandas.read_sql(sql, conn)
    cur.close()
    conn.close()

    return words

  """
  TBD
  # Get phrases per page
  # split terms into phrases
  ps = []
  for gram in grammar:
    gram = gram.replace("'","")
    for term in terms:
      try:
        ps.append("'" + term + " " + gram + "'")
      except:
        print("are your terms and/or grammar actually strings?")

  # Look up phrases
  phrases = pandas.DataFrame()

  for phrase in ps:
    sql = 'SELECT url, source, domain, date, count(count) as count FROM (SELECT url, source, domain, date, regexp_matches(body_text, '+ phrase +', \'g\') as count FROM site_text) t GROUP BY url, source, domain, date' # Does this count # or urls or matches of phrases?
    results = pandas.read_sql(sql, conn)
    results["word"] = phrase.strip("'")
    phrases = phrases.append(results)
    
  phrases

  # Reshape the db outputs for analysis
  ## Merge phrase counts and word counts
  df = words.append(phrases)
  ## Transpose - See: https://stackoverflow.com/questions/52541982/pandas-transpose-one-column
  res = df.pivot(index=['url', 'date', 'domain', 'source'], columns='word', values='count')
  res.reset_index(inplace=True)
  ## Temporary adjustment renaming url as link for later analysis
  #res.rename(columns = {"url": "link"}, inplace=True)

  # Optional: add missing terms. If there are terms with no counts on any pages, they won't get queried, but they will be added here
  #for t in terms:
  #  if t not in list(res.columns):
  #    res[t] = 0
    
  # Add missing pages. If there are pages with no counts for any terms, they won't get queries, but they will be added here
  ## Analyses depend on the full set of pages being present
  ## First, get list of links
  links = list(res["link"].unique())
  links_string = "("
  for l in links:
    l = "'"+l+"',"
    links_string += l
  links_string = links_string[:-1]
  links_string += ")"
  ## Get other pages
  sql = 'SELECT url, date, domain, source FROM site_text WHERE url not in '+links_string
  other_pages = pandas.read_sql(sql, conn)
  other_pages.rename(columns={"url": "link"}, inplace=True)
  ## Combine
  res = res.append(other_pages)
  ## Replace NaNs with Zeros. It's appropriate to do so here because we have scraped the pages and our queries did not return hits.
  res = res.fillna(0)
  ## Reset index
  res.reset_index(drop=True,inplace=True)

  return res
  """

  print("Done!")


"""### Ground Truth / Results"""

# Ground Truth
def ground_truth(pages, fraction, num_terms = None, which_terms = None):
  #random sample of pages, terms
  import random

  term_ground_truth = pages.sample(frac = fraction)
  if num_terms is not None:
    ts = list(random.sample(terms, num_terms)) # Random terms
  else:
    ts = which_terms
  ts.append("link") # Random terms plus other metadata....

  term_ground_truth = term_ground_truth[ts] 
  return term_ground_truth


""" ## Analyze Term Search Results


*  analyze_orgs: calulate and visualize averages and frequencies for each term (Orgs only)
*  analzye_term_correlations: Calculate and visualize co-variance metrics (e.g. AI / digital conservation)
*  co_occurrence: find specific pages with co-occurrence of terms


"""

# Calculate average and frequency per organization
def analyze_orgs(counts, orgs = None, terms = None):
  # Calculate average use and frequency of each term for each org
  
  import pandas

  # Load data
  if type(counts) == str:
    counts = pandas.read_csv(counts) # Load data or use counts generated from scraping above...
  else:
    counts = counts

  # Filter to orgs
  counts = counts.loc[(counts["source"] == "Crawl") & (counts["domain"].isin(orgs))]

  # Prep results
  avg = pandas.DataFrame(index = orgs, columns = [t+"_avg" for t in terms])
  freq = pandas.DataFrame(index = orgs, columns = [t+"_freq" for t in terms])

  for org in orgs:
    df = counts.loc[counts["domain"] == org] # Filter
    ## Average = sum of that term in a given year divided by the total number of URLs for that year, then multiplied by 100. This yields the average use of the term per 100 pages.
    ## Is the term used a lot or a little?
    ## How much the term is used (but not necessarily its distribution - could be used lots on one page)
    for t in terms:
      ## Average
      average = round((df[t].sum() / df[t].count()) * 100, 2)
      avg.loc[avg.index == org, t+"_avg"] = average
      ## Mentioned
      g = df[['url', t]] # Subset the data to just the counts of this term and links
      m = g.loc[g[t]>0] # Create another subset focusing on just the pages where the term was mentioned at least once
      m = m[t].count() # Count the number of pages (links) the term was mentioned at least once
      ## Total
      f = round(( m / df[t].count() ) * 100, 2)
      freq.loc[freq.index == org, t+"_freq"] = f

  return avg, freq

# Calculate co-variance and visualize scatter plots
def analyze_term_correlations(counts, terms = None):
  import pandas
  # Load data
  if type(counts) == str:
    counts = pandas.read_csv(counts) # Load data
  else:
    counts = counts

  # Summarize number of pages
  total_sum = counts['url'].nunique() # Count the total number of pages we're examining.
  #print(total_sum) # Debugging

  # Set up terms
  for t in terms: # For each term in our list of terms. Or, pick a specific term by doing for t in terms[terms.index("YOUR TERM")]
    # make sure the term has actually been counted here?
    t_count = counts[[t]] # The counts of this term
    for ot in terms: # For each of the other terms in our list
      if t != ot: # Don't count the term against itself
        ot_count = counts[[ot]] # The counts of this other term
        #display(ot_count, t_count) # Debugging
        joined = t_count.join(ot_count, lsuffix="_term", rsuffix="_otherterm") # The two (counts of this term, plus this other term) together
        #display(joined[1:,]) # Debugging
        c = list(joined.columns)
       
        # Calculate some metrics
        zerozeros = joined.loc[(joined[c[0]] == 0) & (joined[c[1]] == 0)] # Share of 0,0s. Example: Neither "Machine Learning" nor "Conservation"
        thiszero = joined.loc[(joined[c[0]] == 0) & (joined[c[1]] > 0)] # Share of this term 0s, other term at least once. No "Machine Learning" but at least one "Conservation". Gives us a sense of whether the terms are being discussed together.
        thatzero = joined.loc[(joined[c[0]] > 0) & (joined[c[1]] == 0)] # Share of other term 0s, this term at least once. No "Conservation" but at least one "Machine Learning".  Gives us a sense of whether the terms are being discussed together.
        together = joined.loc[(joined[c[0]] > 0) & (joined[c[1]] > 0)] # Share of >1, >1. "Machine Learning" and "Conservation" used together
        
        print(t, ot) # Could save instead of printing this information....
        print("Zeros: {}, {}%".format( zerozeros.shape[0], int((zerozeros.shape[0] / total_sum) * 100))) # Percent and count of all pages these terms never appear on together
        print("{} but not {}: {}, {}%".format(ot, t, thiszero.shape[0], int((thiszero.shape[0] / total_sum) * 100))) # Percent and count of pages where the other term appears, but not this term
        print("{} but not {}: {}, {}%".format(t, ot, thatzero.shape[0], int((thatzero.shape[0] / total_sum) * 100))) # Percent and count of pages where this term appears, but not the other one
        print("Together: {}, {}%".format( together.shape[0], int((together.shape[0] / total_sum) * 100))) # Percent and countof pages these terms appear on together
        print("\n")
        
        # Calculate basic correlation. Assumption - when mentioned together, terms should be more or less equally mentioned, increasing or decreasing together linearally. Otherwise, we're just talking about, for instance, "Machine Learning" but not in the context of "Conservation" or vice versa.
        correlation = together[[c[0], c[1]]].corr() # https://towardsdatascience.com/statistics-in-python-understanding-variance-covariance-and-correlation-4729b528db01
        display(correlation)
        
        # Visualize
        display(together.plot.scatter(x = c[0], y = c[1])) # Show scatter plot. Replace joined with thiszero to show no "this term" but at least one "other term". Replace with together to show pages where both used.

# Where are organizations talking about terms together? Once we know that, we can code for *how* they talk about terms in specific depth        
def co_occurrence(words, terms, orgs = False):
  # Filter words to > 0
  filtered = words.copy()
  for term in terms:
    filtered = filtered.loc[filtered[term] > 0]

  # Filter to any orgs
  if orgs != False:
    for org in orgs:
      filtered = filtered.loc[filtered["domain"] == org]

  # Export results
  t = ""
  for term in terms:
    t += term+"_"
  t += ".csv"
  filtered = filtered[["url", "domain", "source", "date"] + terms]
  filtered.to_csv(t)

  return filtered
