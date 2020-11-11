text = '''After seven years, the Microsoft Xbox and Sony PlayStation game consoles are getting big upgrades.

The companies are releasing their next-generation consoles just in time for the holiday shopping season. Microsoft will release two models on Tuesday: the Xbox Series X, the $500 version, and the Xbox Series S, its $300 lower-powered sibling. Sony will release its PlayStation 5, which comes in two models for $400 and $500, next Thursday.

So which one will you choose? There are many different types of gamers out there, so we — Brian X. Chen, a longtime PlayStation fan, and Mike Isaac, who grew up playing the Xbox — both tested the new consoles in our homes.

This review focuses on the new Xbox systems. Our review of the PlayStation 5 will follow this week.

BRIAN Hello, Mike. While the country has been tallying up votes for our next president, you and I have been playing video games to help our readers decide which new game console to vote for with their wallets.

For the last generation of consoles, PlayStation 4 was the must-have game device, with more than double the number of sales of the Xbox One. Now people are wondering if it’s the Xbox’s turn to win with its sleek, rectangular Series X and Series S.

What are your impressions so far?

MIKE For quite some time I was an Xbox loyalist. I remember back when I was in high school and the first Halo game came out. It was a must-play game, one of the best shooters of its time. Its success made owning an Xbox a priority.

Now, nearly 20 years later, I don’t have that same feeling with the Xbox Series X and Series S. There’s not an exclusive, Xbox-only game that I’m generally hyped up about, you know?

Thanks for reading The Times.
Subscribe to The Times
BRIAN Well, the new Halo game, called Halo Infinite, was supposed to be the big launch title to market these new Xbox systems. It’s a big disappointment that the director of the game stepped down and the project was delayed.

MIKE It’s a huge, noticeable absence, especially when you’re trying to launch a competing product to Sony — whose PlayStation 4 has dominated the market for the past seven years.

I’ll give it this: Hardware-wise, the Series X has many similarities to Sony’s new PlayStation. They both include solid-state drives, a storage technology that loads games faster than traditional spinning hard-disk drives.

Editors’ Picks

How Do You Know When Society Is About to Fall Apart?

A Guaranteed Monthly Check Changed His Life. Now He Sends Out 650.


BRIAN The new Xboxes and PlayStations also have graphics processors that support “ray tracing,” which is a complex graphics rendering process that makes lighting and shadows look more realistic. That in turn translates to much better graphics.

MIKE The Xbox controller retains the classic shape of what you’re used to with an Xbox, but it is trimmed down and sleeker — a match with the more elegant, minimalist design of the new Xbox models.


ImageThe Xbox controller is trimmed down and sleek.
The Xbox controller is trimmed down and sleek.Credit...Brandon Ruffin for The New York Times
But here’s the thing: If the systems are on fairly level footing, technology-wise, it makes the game releases themselves that much more important.

BRIAN I agree. In terms of hardware features compared with the PlayStation, the Xbox is a tiny bit better. The Series X includes about 20 percent more storage for holding downloaded games than the PlayStation 5. The console is a compact tower that will be easier to fit into an entertainment center than the bulky PlayStation 5.

But that edge is negligible without killer games to play. For now, there isn’t anything all that compelling. Launch titles include Assassin’s Creed: Valhalla, Gears Tactics and Yakuza: Like a Dragon, among others. I tried a handful of launch games, and the graphics looked great and the console felt fast, but the games were not appealing to me.
'''

from summa.summarizer import summarize
from summa import keywords

# Define length of the summary as a proportion of the text

print(summarize(text, words=100))
#summarize(text, words=50)

#print("Keywords:\n", keywords.keywords(text))
print("Top 3 Keywords:\n", keywords.keywords(text, words=3))

