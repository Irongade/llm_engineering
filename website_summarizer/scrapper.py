from bs4 import BeautifulSoup

import requests 

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

class WebScrapper:
    def __init__(self, url) -> None:
        self.fetch_content(url)

    def fetch_content(self, url):
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")

        self.content: BeautifulSoup = soup

        return

    def fetch_website_contents(self):
        """
        Return the title and contents of the website at the given url;
        truncate to 2,000 characters as a sensible limit
        """

        title = self.content.title.string if self.content.title else "No title found after parsing"

        if self.content.body:
            for irrelevant in self.content.body(["script", "style", "img", "input"]):
                irrelevant.decompose()
            text = self.content.body.get_text(separator="\n", strip=True)
        else:
            text = ""

        return (title + "\n\n" + text)[:2_000]

    def fetch_website_links(self):
        """
        Return all the links on the webiste at the given url
        """

        links = [link.get("href") for link in self.content.find_all("a")]

        return [link for link in links if link]
