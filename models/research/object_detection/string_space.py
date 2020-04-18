# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 02:04:53 2020

@author: Navin Subbu
"""

import string
s = "how much for the maple syrup? $20.99? That's ricidulous!!!"
y = "how much for the maple syrup? $20.99? That's ricidulous!!!"
for char in string.punctuation:
    s = s.replace(char, ' ')
    
print(s)    


whitelist = string.digits + string.ascii_letters + ' '
new_s = ''
for char in y:
    if char in whitelist:
        new_s += char
    else:
        new_s += ''
        
print(new_s)