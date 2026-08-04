import json
from website_summarizer import Model, SeleniumWebScrapper, ChatOptions
from IPython.display import Markdown, display, update_display

class BrochureGenerator:
    def __init__(self) -> None:
        # uses default OpenAI Model
        self.content_model = Model(model="gpt-5-nano")
        self.brochure_model = Model(model="gpt-4.1-mini")

        
    def _get_link_system_prompt(self) -> str:
        return """
            You are provided with a list of links found on a webpage.
            You are able to decide which of the links would be most relevant to include in a brochure about the company,
            such as links to an About page, or a Company page, or Careers/Jobs pages.
            You should respond in JSON as in this example:
            {
                "links": [
                    {"type": "about page", "url": "https://full.url/goes/here/about"},
                    {"type": "careers page", "url": "https://another.full.url/careers"}
                ]
            }
        """
    
    def _get_links_user_prompt(self, url: str) -> str:
        user_prompt = f""" 
            Here is the list of links on the website {url} -
            Please decide which of these are relevant web links for a brochure about the company, 
            respond with the full https URL in JSON format.
            Do not include Terms of Service, Privacy, email links.

            Links (some might be relative links):
        """
        scrapper = SeleniumWebScrapper(url)
        links = scrapper.fetch_website_links()
        print(len(links), links)
        user_prompt += "\n".join(links)

        return user_prompt

    def select_relevant_links(self, url: str):
        messages = [
            {"role": "system", "content": self._get_link_system_prompt()},
            {"role": "user", "content": self._get_links_user_prompt(url)}
        ]

        options = ChatOptions(response_format={"type": "json_object"}, stream=None)
        result = self.content_model.chat(messages=messages, options=options)
        links = json.loads(result)

        return links

    def fetch_page_and_all_relevant_links(self, url: str) -> str:
        scrapper = SeleniumWebScrapper(url)
        contents = scrapper.fetch_website_contents()
        relevant_links = self.select_relevant_links(url)

        result = f"## Landing Page: \n\n{contents}\n ## Relevant Links: \n"

        for link in relevant_links["links"]:
            link_scrapper = SeleniumWebScrapper(link['url'])
            result += f"\n\n ### Link: {link['type']}\n"
            result += link_scrapper.fetch_website_contents()
        
        return result


    def get_brochure_system_prompt(self) -> str:
        return """
            You are an assistant that analyzes the contents of several relevant pages from a company website
            and creates a short brochure about the company for prospective customers, investors and recruits.
            Respond in markdown without code blocks.
            Include details of company culture, customers and careers/jobs if you have the information.
        """

    def get_brochure_user_prompt(self, company_name, url) -> str:
        user_prompt = f""" 
            You are looking at a company called: {company_name}
            Here are the contents of its landing page and other relevant pages;
            use this information to build a short brochure of the company in markdown without code blocks.\n\n
        """

        user_prompt += self.fetch_page_and_all_relevant_links(url)
        user_prompt = user_prompt[:5_000] 
        return user_prompt

    def create_brochure(self, company_name, url, stream = False) -> str:
        messages = [
            {"role": "system", "content": self.get_brochure_system_prompt()},
            {"role": "user", "content": self.get_brochure_user_prompt(company_name, url)}
        ]

        options = ChatOptions(stream=stream, response_format=None)

        if stream:
            response = ""
            display_handle = display(Markdown(response), display_id=True)
            for chunk in self.brochure_model.chat_stream(messages=messages, options=options):
                response += chunk
                update_display(Markdown(response), display_id=display_handle.display_id)
        else:
            result = self.brochure_model.chat(messages=messages, options=options)
            display(Markdown(result))

print("\n")

bg = BrochureGenerator()

bg.create_brochure("HuggingFace", "https://huggingface.co")