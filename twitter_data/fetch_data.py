import twint

import os

c = twint.Config()
# c.Search = "#polska OR #polityka lang:pl"
c.Search = "#polityka lang:pl"
c.Since = '2019-11-16'
c.Store_csv = True
c.Tabs = True
c.Hide_output = True
c.Output = os.path.join('twitter_data', 'data', 'polityka.csv')
twint.run.Search(c)
