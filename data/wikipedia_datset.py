import wikipediaapi 
import os 


class Wikipedia_Dataset(): 
    def __init__(self):
        self.wikipedia = wikipediaapi.Wikipedia('en')
        self.categories = ["Category:Physics", "Category:Countries", "Category:Books"]
        self.categories_articles = self.get_categories_articles()

    def get_categorymembers(self, categorymembers, level=0, max_level=1):
        categories_titles = []
        for cm in categorymembers.values():
            categories_titles.append(cm.title)
            if cm.ns == wikipediaapi.Namespace.CATEGORY and level<max_level :
                categories_titles.extend(self.get_categorymembers(cm.categorymembers, level=level+1, max_level=max_level))
        return categories_titles 
    
    def extract_content_pages(self, files, page_name, languages):
        # iterate over languages
        for lang in languages:
            try:
                self.wikipedia = wikipediaapi.Wikipedia(language=lang, extract_format=wikipediaapi.ExtractFormat.WIKI)
            except:
                continue
            try:
                files[lang]
            except KeyError:
                files[lang] = {}

            try:
                files[lang][page_name] = self.wikipedia.page(page_name).text
            except:
                continue
        return files
    
    def store_results(self, c, title, files, languages):
        if not os.path.exists('wikipedia_articles'):
            os.makedirs('wikipedia_articles')
        for lang in languages:
            if files[lang][title] == "":
                #print(True)
                continue
            if not os.path.exists(os.path.join('wikipedia_articles',f'{c}', f'{lang}')):
                os.makedirs(os.path.join('wikipedia_articles',f'{c}', f'{lang}'))
            with open((os.path.join('wikipedia_articles',f'{c}', f'{lang}', f'{title.lower().replace(" ", "-").replace("/", "")}.txt')), 'w') as file:
                #print(lang)
                #print(title)
                #print(files[lang][title])
                file.write(files[lang][title])
        

    def get_categories_articles(self):
        categories_articles = {}
        for c in self.categories :
            self.wikipedia = wikipediaapi.Wikipedia('en')
            c_page = self.wikipedia.page(c)
            c_titles = self.get_categorymembers(c_page.categorymembers)
            categories_articles[c] = []
            for title in c_titles:
                page_t = self.wikipedia.page(title)
                files = {}
                languages = list(page_t.langlinks.keys())
                languages.append('en')
                articles = self.extract_content_pages(files,title, languages)
                self.store_results(c,title,files, languages)
                categories_articles[c].append(files)  #--------> This is to be updated 
               
        return categories_articles


