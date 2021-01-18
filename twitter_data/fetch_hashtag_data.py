import twint
import os

hashtags = [
    '#LGBTtoLudzie',
    '#LGBTtoIdeologia',
    '#IdeologiaLGBT',
    '#StopAgresjiLGBT',
    '#homofobia',
    '#BabiesLifesMatter',
    '#piekłodzieci',
    '#AborcjaBezGranic',
    '#reżimPiS',
    '#ulgaabolicyjna',
    '#500plus',
    '#renty',
    '#emerytury',
    '#płacaminimalna',
    '#wynagrodzenia',
    '#firmy',
    '#pracownicy',
    '#socjalizm',
    '#własność',
]

for tag in hashtags:
    c = twint.Config()
    c.Since = '2019-11-16'
    c.Search = f'({tag}) lang:pl'
    c.Store_csv = True
    c.Tabs = True
    c.Output = os.path.join('twitter_data', 'data', f'{tag}.csv')
    c.Hide_output = True
    twint.run.Search(c)
