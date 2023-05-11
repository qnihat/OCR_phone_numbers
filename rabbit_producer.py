import pika

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost'))
channel = connection.channel()
def create_mq_queue(queue_name):
    channel.queue_declare(queue=queue_name)
    print('Queue created: ', queue_name)

def publish_message_to(que_name,messaj):
    channel.basic_publish(exchange='', routing_key=que_name, body=messaj)
    #connection.close()