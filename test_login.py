from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import time

service = Service("C:\\Users\\Admin\\Downloads\\chromedriver-win32\\chromedriver-win32\\chromedriver.exe")

driver = webdriver.Chrome(service=service)
driver.get("https://www.saucedemo.com/")
time.sleep(2)

driver.find_element("id", "user-name").send_keys("standard_user")
driver.find_element("id", "password").send_keys("secret_sauce")
driver.find_element("id", "login-button").click()
time.sleep(2)

if "inventory" in driver.current_url:
    print("✅ Login successful!")
else:
    print("❌ Login failed.")

driver.quit()
