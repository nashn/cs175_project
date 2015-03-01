# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 11:36:32 2015

@author: nash
"""

#xml_parser.py

import xml.etree.ElementTree as ET
tree = ET.parse('./datasets/spam.xml')
root = tree.getroot()

print( len(root) )

'''
for child in root:
    print( child.tag, ' --- ', child.attrib )
'''    
f = open('./datasets/spam_cleanup.txt', 'a', encoding='utf-8')



for sms in root.findall('sms'):
    c = sms.find('class').text
    source = sms.find('source').text
    text = sms.find('text').text

    f.write( text + '\n\n' )    
    print( source )
    
f.close()
    