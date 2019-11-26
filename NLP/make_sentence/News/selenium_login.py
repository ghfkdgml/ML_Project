from selenium import webdriver

USER="suho1898"
PASS="rlatngh1898-"

browser = webdriver.PhantomJS()
browser.implicitly_wait(3)

url_login = "https://nid.naver.com/nidlogin.login?mode=form&url=https%3A%2F%2Fwww.naver.com"
browser.get(url_login)
print("go to login page")

e = browser.find_element_by_id("id")
e.clear()
e.send_keys(USER)
e = browser.find_element_by_id("pw")
e.clear()
e.send_keys(PASS)

form = browser.find_element_by_css_selector("input.btn_global[type=submit]")
try:
    form.submit()
    print("login")
except Exception as e:
    print(e)

try:
    browser.get("https://mail.naver.com/")
    name = browser.find_element_by_css_selector(".gnb_name")
    print(name.text)
except Exception as e:
    print(e)
