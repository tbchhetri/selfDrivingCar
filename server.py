#creates a local network
import socket
import threading #enables to establish multiples connections with different clients

HEADER = 64 #the length of the message
PORT = 5050
#SERVER = "192.168.0.4"
SERVER = socket.gethostbyname(socket.gethostname()) #this will automatically get the ip address
print(SERVER)
print(socket.gethostname()) #gets the computer's name
ADDR = (SERVER, PORT)
FORMAT = 'utf-8' #the protocol used for decoding
DISCONNECT_MESSAGE = "!DISCONNECT"

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #socket.socket is for making new socket, socket.AF_INET (internet) is the type of socket we are looking for, there are others too like bluetooth, lastly SOCK_STREAM is saying we will be streaming data
server.bind(ADDR) #binds our socket to this address

#listning
def handle_client(connection, addr):
    print(f"[NEW CONNECTION] {addr} connected.")

    connected = True
    while connected:
        msg_length = connection.recv(HEADER).decode(FORMAT) #so this takes the number of bytes to be recieved from the client, but since we don't know how many we make use of a protocol

        if msg_length: #if the message is not empty
            msg_length = int(msg_length) #finds the length of message coming
            msg = connection.recv(msg_length).decode(FORMAT)

            #handeling the disconnection
            if msg == DISCONNECT_MESSAGE:
                connected = False

            print(f"[{addr}] {msg}")
            connection.send("Msg recieved".encode(FORMAT))
    connection.close()

def start ():
    server.listen()
    print(f"[LISTNING Server is listening on {SERVER}")

    while True:
        connection, addr = server.accept()  #addr stores the ip address and the port of the client, connection is the connection object that allows to talk back to the clients
        thread = threading.Thread(target=handle_client, args=(connection, addr)) #notice we are calling the function and passing the argument in this wierd format
        thread.start()
        print(f"[ACTIVE CONNECTIONS] {threading.activeCount() - 1}") #the amount of client connected

print("[STARTING] server is starting...")
start()
