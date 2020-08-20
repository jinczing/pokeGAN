# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
import scrapy
import re
from itemadapter import ItemAdapter
from scrapy.exceptions import DropItem
from scrapy.pipelines.images import ImagesPipeline

class PokePipeline(ImagesPipeline):
	abs_img_path = 'D:/projects/pokeGAN/pokemon_images/'

	def file_path(self, request, response=None, info=None):
		#return  abs_img_path + re.search(r'[0-9]+',request.url.split('/')[-1]).group(0) + '.jpg'
		return 'full/' + re.search(r'[0-9]+',request.url.split('/')[-1]).group(0) + '.png'

	def get_media_requests(self, item, info):
		for image_url in item['image_urls']:
			yield scrapy.Request(image_url)


