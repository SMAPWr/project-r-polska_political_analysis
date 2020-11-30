import twint

import os

pis = [
    "BeataSzydlo",
    "jbrudzinski",
    "Macierewicz_A",
    "mblaszczak",
    "MariuszKaminsk",
    "MarekKuchcinski",
    "StKarczewski",
    "RyszardTerlecki",
    "r_czarnecki",
    "RzecznikPiS",
    "PiotrGlinski",
    "marekgrobarczyk"
    "anita_cz",
    "profKarski",
    "ZbigniewKuzmiuk",
    "beatamk",
    "TomaszPoreba",
    "E_Rafalska",
    "mareksuski",
    "elzbietawitek"
]

po = [
    "bbudka",
    "Arlukowicz",
    "EwaKopacz",
    "TomaszSiemoniak",
    "trzaskowski_",
    "MKierwinski",
    "M_K_Blonska",
    "jacek_protas",
    "AndrzejHalicki",
    "TomaszLenz",
    "SlawomirNitras",
    "CTomczyk"
]

lewica = [
    "KGawkowski",
    "mmzawisza",
    "wieczorekdarek",
    "MagdaBiejat",
    "B_Maciejewska",
    "KPawliczak",
    "Pawlowska_pl",
    "poselTTrela",
    "K_Smiszek",
    "AnitaKDZG",
    "PolaMatysiak",
    "AM_Zukowska",
    "wlodekczarzasty",
    "GabrielaMorStan"
]

konederacja = [
    "Kulesza_pl",
    "krzysztofbosak",
    "JkmMikke",
    "RobertWinnicki",
    "GrzegorzBraun_",
    "KonradBerkowicz",
    "ArturDziambor",
    "D_Sosnierz",
    "TudujKrzysztof",
    "urbaniak_michal",
    "K_Kaminski_",
]

psl = [
    "KosiniakKamysz",
    "Paslawska",
    "DariuszKlimczak",
    "JarubasAdam",
    "StruzikAdam",
    "PZgorzelskiP",
    "AwRakoczy",
    "GrzybAndrzej",
    "BrzezinMarek",
    "KasprzakMiecz",
    "PawelGancarz",
    "StefanKrajewski",
    "Jarek_Rzepa",
    "MackowiakJanusz",
    "CzSiekierski",
    "SawickiMarek",
    "TPilawka",
]

parties = {'pis': pis, 'po': po, 'lewica': lewica,
           'konfederacja': konederacja, 'psl': psl}

# os.mkdir('data')
# for p in parties:
#     os.mkdir(os.path.join('data', p))

for p, users in parties.items():
    for user in users:
        c = twint.Config()
        c.Username = user
        c.Since = '2019-11-16'
        c.Store_csv = True
        c.Tabs = True
        c.Output = os.path.join('data', p, f'{user}.csv')
        twint.run.Search(c)
