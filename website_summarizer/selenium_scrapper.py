from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import StaleElementReferenceException

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

class SeleniumWebScrapper:
    def __init__(self, url: str, wait = 10) -> None:
        self.url = url
        self.content = self.fetch_content(url, wait)
    
    def initialiseOptions(self) -> Options:
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--window-size=1920, 1080")
        options.add_argument(f"--user-agent={USER_AGENT}")

        return options

    def fetch_content(self, url, wait = 10) -> BeautifulSoup:
        options = self.initialiseOptions()

        driver = webdriver.Chrome(options=options)

        try:
            driver.set_page_load_timeout(wait)
            driver.get(url)

            self.title = driver.title or "No title found"
            self.text = driver.execute_script("return document.body.innerText") or ""

            all_links = driver.find_elements(By.TAG_NAME, "a")
            # hrefs = [link.get_attribute("href") for link in all_links]
            # self.links = [
            #     href
            #     for href in hrefs if href
            # ]
            hrefs = []
            for link in all_links:
                try:
                    href = link.get_attribute("href")
                    if href:
                        hrefs.append(href)
                except StaleElementReferenceException:
                    continue
            self.links = hrefs

        finally:
            driver.quit()

    def fetch_website_contents(self, limit = 2_000):
        """
        Return the title and contents of the website at the given url;
        truncate to 2,000 characters as a sensible limit
        """

        return (self.title + "\n\n" + self.text)[:limit]

    def fetch_website_links(self):
        """
        Return all the links on the webiste at the given url
        """

        return self.links
