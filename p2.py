import re
import whois

def category2(website):
        
    file_obj = open(r'phishing5.txt','a')
    domain_reg_len = 300 #globally initialized
    try:
    #8 Domain Registration Length -> Domain Registration Length means for the number of year's, the domain renewal amount paid in advance to the Registrar. The website domain should not be expired
        page = whois.whois(website) #whois -> amount of time/date of a websites domain (google.com)
        if type(page.expiration_date) == list: #list -> type of expiration_date like (2020,15 may to 2020,20 may ) will come in the list form
            domain_reg_len = (page.expiration_date[0] - page.creation_date[0]).days #) index will tell the year of the date and then convert it to days 
        else:
            domain_reg_len = (page.expiration_date - page.creation_date).days # if only the years then no indexing required subtract the number of years 
    #print domain_reg_len
        if domain_reg_len <= 365:
            file_obj.write('-1,') #phishing
        else:
            file_obj.write('1,') #legitimate
    except:
        file_obj.write('-1,') 
    #9 Using Non-Standard Port 
    match_port = re.search(':[//]+[a-z]+.[a-z0-9A-Z]+.[a-zA-Z]+:([0-9#]*)',website) #standard port number like mongodb has 27017
    if match_port:
        print (match_port.group()) #releasing tuple of the regex 
        if match_port.group(1) == '#': 
            file_obj.write('-1,')
        else:
            file_obj.write('1,')
    else:
        file_obj.write('1,')
    file_obj.close()


