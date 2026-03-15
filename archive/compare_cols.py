#!/usr/bin/env python3
"""Compare Excel columns to model requirements."""

# What we HAVE from the Excel
have = ['Team','Conference','NET Rank','PrevNET','AvgOppNETRank','AvgOppNET',
        'WL','ConfWL','ConfSOS','NCWL','RoadWL','NETSOS','NETNCSOS',
        'WABRk','WAB','NCWABRk','NCWAB','Q1','Q2','Q3','Q4']

# What our model NEEDS
need = ['RecordID','Season','Team','Conference','Overall Seed','Bid Type',
        'NET Rank','PrevNET','AvgOppNETRank','AvgOppNET',
        'WL','Conf.Record','Non-ConferenceRecord','RoadWL',
        'NETSOS','NETNonConfSOS','Quadrant1','Quadrant2','Quadrant3','Quadrant4']

print('MODEL NEEDS               EXCEL HAS')
print('='*55)
mapping = {
    'RecordID':              '(generate: 2025-26-TeamName)',
    'Season':                '(constant: 2025-26)',
    'Team':                  'Team  YES',
    'Conference':            'Conference  YES',
    'Overall Seed':          '(leave blank, predicted)',
    'Bid Type':              'MISSING (need bracketology)',
    'NET Rank':              'NET Rank  YES',
    'PrevNET':               'PrevNET  YES',
    'AvgOppNETRank':         'AvgOppNETRank  YES',
    'AvgOppNET':             'AvgOppNET  YES',
    'WL':                    'WL  YES',
    'Conf.Record':           'ConfWL  YES (rename)',
    'Non-ConferenceRecord':  'NCWL  YES (rename)',
    'RoadWL':                'RoadWL  YES',
    'NETSOS':                'NETSOS  YES',
    'NETNonConfSOS':         'NETNCSOS  YES (rename)',
    'Quadrant1':             'Q1  YES (rename)',
    'Quadrant2':             'Q2  YES (rename)',
    'Quadrant3':             'Q3  YES (rename)',
    'Quadrant4':             'Q4  YES (rename)',
}
for col in need:
    status = mapping.get(col, '???')
    print(f'  {col:<25} -> {status}')

print()
print('EXTRA columns in Excel (bonus, not required):')
print('  ConfSOS, WABRk, WAB, NCWABRk, NCWAB')
