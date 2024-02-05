from bs4 import BeautifulSoup

def process_div(html):
    table = []
    for div in reversed(list(html.children)):
        games = [child for child in div.children if child.name == 'div' and 'game' in child.get('class', [])]
        
        for game in reversed(list(games)):
            server_wrapper = next((child for child in game.children if child.name == 'div' and 'team1' in child.get('class', [])), None) or \
                            next((child for child in game.children if child.name == 'div' and 'team2' in child.get('class', [])), None)
            server_value = -1 if 'team1' in server_wrapper.get('class', []) else 1
            for point in reversed(list((child for child in game.children if child.name == 'div' and 'point' in child.get('class', [])))):
                for divs in (child for child in point.children if child.name == 'div'):
                    for div in (child for child in divs.children if child.name == 'div' and 'pointScore' in child.get('class', [])):
                        point_score = div.text
                        if '-' not in point_score:
                            continue
                        a, b = point_score.split(' - ')
                        table.append([a, b, server_value])
    return table

with open('./assets/doc/html1.html', 'r') as f:
    html_content = f.read()

soup = BeautifulSoup(html_content, 'html.parser')
for child in soup.children:
    table = process_div(child)

print(len(table))

with open ('./assets/doc/output.csv', 'w') as f:
    for row in table:
        for i in range(len(row)):
            if(row[i] == 'AD'):
                row[i] = '50'
            print(row[i], end='',file=f)
            if(i != len(row) - 1):
                print(' ', end='',file=f)
        print(file=f)
