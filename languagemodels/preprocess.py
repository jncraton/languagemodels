from html import unescape
from html.parser import HTMLParser


def get_html_paragraphs(src: str):
    """
    Return plain text paragraphs from an HTML source

    This function is designed to be quick rather than robust.

    It follows a simple approach to extracting text:

    1. Ignore all content inside the following elements listed in `ignore`.
    2. Merge inline text content into paragraphs from `inlines` set.
    3. Convert any newly merged text element with at least `min_length`
    characters to a paragraph in the output text.

    >>> get_html_paragraphs(open("test/wp.html").read())
    'Bolu Province (Turkish: Bolu ili) is a province...'

    >>> get_html_paragraphs(open("test/npr.html").read())
    "First, the good news. Netflix reported a record ..."
    """

    class Element:
        def __init__(self, tag, text):
            self.tag = tag
            self.text = unescape(text)

    elements = []

    class MyHTMLParser(HTMLParser):
        ignoring = False
        ignore = ("script", "style", "header", "footer")
        inlines = ("a", "b", "i", "span", "sup", "sub", "strong", "em")

        def handle_starttag(self, tag, attrs):
            if tag in self.ignore:
                self.ignoring = True

            if not self.ignoring and tag not in self.inlines:
                elements.append(Element(tag, ""))

        def handle_endtag(self, tag):
            if tag in self.ignore:
                self.ignoring = False

            if not self.ignoring and tag not in self.inlines:
                elements.append(Element("/" + tag, ""))

        def handle_data(self, data):
            if not self.ignoring:
                if elements and elements[-1].text:
                    elements[-1].text += data
                else:
                    elements.append(Element(None, data))

    parser = MyHTMLParser()
    parser.feed(src)

    text = ""

    for el in elements:
        if not el.tag and len(el.text) > 140:
            text += el.text.strip() + "\n\n"

    return text.strip()
