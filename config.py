RABBIT_QUEUE_NAME='handwritten_numbers'
FOLDERS_TO_SAVE_NUM_IMGS='img/extracted_numbers/'
SOURCE_IMAGES='img/images_from_milana/'

#rabbit docker create cmd
#docker run --rm -it -p 15672:15672 -p 5672:5672 rabbitmq:3-management