# This package will contain the spiders of your Scrapy project
#
# Please refer to the documentation for information on how to create and manage
# your spiders.
import scrapy
import os
import re
from .image_item import ImageItem

# site to pokemon wiki
site_url = '''https://wiki.52poke.com/zh-hant/%E5%AE%9D%E5%8F%AF%E6%A2%A6%E5%88%97%E8%A1%A8%EF%BC%88%E6%8C%89%E5%85%A8%E5%9B%BD%E5%9B%BE%E9%89%B4%E7%BC%96%E5%8F%B7%EF%BC%89'''
attribute_path = 'D:/projects/pokeGAN/pokemon_attributes'

stat_dict = ['tr.bgl-HP', 'tr.bgl-攻击', 'tr.bgl-特攻', 'tr.bgl-防御', 'tr.bgl-特防', 'tr.bgl-速度']
type_dict = {'一般': 0, '火':1, '水':2, '電':3, '电':3, '草':4, '冰':4, '格鬥':5, '格斗':5, '毒':6, '地面':7
		, '飛行': 8,  '飞行': 8,'超能力': 9, '蟲':10, '虫':10, '岩石':11, '幽靈':12, '幽灵':12, '龍':13, '龙':13
		, '惡':14, '恶':14, '钢':15, '鋼':15, '妖精':16}

class PokeSpider(scrapy.Spider):
	name = 'poke'
	start_urls = [site_url]

	def parse(self, response):
		tables = response.css('table.roundy.eplist')[:9] # main table in poke wiki

		urls = []
		counter = 0
		for table in tables:
			for tr in table.css('tr')[2:]: # filter header
				counter += 1
				urls.append('https://wiki.52poke.com' + tr.css('td').css('a')[0].attrib['href'])

		for url in urls:
			yield scrapy.Request(url=url, callback=self.parse_each)

	def parse_each(self, response):

		# parse image url
		raw_url = response.css('table.roundy.bgwhite.fulltable')[0].re('data\-url\=\"([^\"]+)\"')[0]
		img_url = 'http:' + raw_url.split('/thumb')[0] + '/'.join(raw_url.split('/thumb')[1].split('/')[:-1]) # url to each image resource

		#pares attribute
		poke_id = re.search(r'[0-9]+',img_url.split('/')[-1]).group(0) # parse poke_id from image url

		ele_types = []
		for q in response.css('span.type-box-8-inner').re(r'title\=\"(.+)\（属性')[:2]:
			try:
				ele_types.append(type_dict[q])
			except:
				print('element type error occurs.')

		height = response.css('td.roundy.bw-1').re(r'([0-9]+\.[0-9]+)m')[0]

		weight = response.css('td.roundy.bw-1').re(r'([0-9]+\.[0-9]+)kg')[0]
		
		stats = []
		for s in stat_dict:
			stats.append(response.css(s).css('div::text').getall()[1])

		# write attribute file, format: pokemon id, types, height, weight, hp, attack, special attack, defense, special defence, speed
		with open(attribute_path + '/' + poke_id + '.txt', 'w', newline='') as f:

			f.write(poke_id + '\n')

			if ele_types:
				for ele_type in ele_types:
					f.write(str(ele_type) + '\n')
			else:
				f.write(str(-1) + '\n')

			f.write(height + '\n')

			f.write(weight)

			for stat in stats:
				f.write('\n' + stat)

		item = ImageItem()
		item['image_urls'] = [img_url]
		return item

