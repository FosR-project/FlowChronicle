
'''columns we never encode as fix values'''
never_fix = ['Src IP Addr', 'Dst IP Addr'] # FIXME

'''
columns that are IP addresses, used for pattern extension
'''
ip_columns = ['Dst IP Addr', 'Src IP Addr']

'''
columns that we can encode as with placeholders, respectivly set placeholders
all combinations of these columns are considered
'''
placeholder_set_columns=['Dst IP Addr', 'Src IP Addr']
placeholder_get_columns=['Dst IP Addr', 'Src IP Addr']
