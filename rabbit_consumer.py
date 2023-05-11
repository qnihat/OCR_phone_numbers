import time
import pika, sys, os
import json
from detect_symbols import detect_sym
from number_arranger import arrabge_numbers
from save_nums_to_file import save_numbers_to_file

from config import RABBIT_QUEUE_NAME
def consume_rabbit(que_name):
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()
    q=channel.queue_declare(queue=que_name)
    #q_len = q.method.message_count
    #print('len of the que:',q_len)

    def callback(ch, method, properties, body):
        message = json.loads(body)

        uuid_name_img=message['file_name']
        dict_of_ = detect_sym(uuid_name_img)
        final_number = arrabge_numbers(dict_of_, uuid_name_img)
        print(final_number)
        save_numbers_to_file(final_number)



    channel.basic_consume(queue=que_name, on_message_callback=callback, auto_ack=True)

    #print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()

def Start_rabbit_consumer():
    if __name__ == '__main__':
        try:
            consume_rabbit(RABBIT_QUEUE_NAME)()
        except KeyboardInterrupt:
            print('Interrupted')
            #print_table_content()
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)

Start_rabbit_consumer()