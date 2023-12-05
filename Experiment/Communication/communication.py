"""
Connects Unity and Python without the use of ML-agents.

Written by Minguk Kim @ Texas A&M University
mingukkim@tamu.edu
"""

import socket
import struct

# TODO: converting timeScale value by using Python.

class UnityCommunication:
    def __init__(self, host="127.0.0.1", port=25001):
        """
        Initializes the UnityCommunication class.

        Parameters:
        - host (str): The IP address of the Unity instance.
        - port (int): The port number to connect to the Unity instance.
        """
        self.host = host
        self.port = port
        self.sock = None
        
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    BELOW ARE METHODS FOR TRAINING (OR INFERENCE)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
    def receive_state_info(self, var_name, varType):
        """
        Sends  (or retrieves) an object's position (or rotation degree) to Unity via TCP.
        
        Parameters:
        - var_name (string): A name of variable that is in the Unity env.
        - varType (string): ...
        """
        try:
            # request info with Unity
            self.sock.sendall(b'STATE ' + var_name.encode('utf-8'))
            
            if varType == "xyz":            

                # Await response from Unity containing updated data.
                response = self.sock.recv(12) # 3 (floats) * 4 (Bytes each) = 12 bytes

                # Convert the byte data back into x, y, and z float values.
                x, y, z = struct.unpack('fff', response)

                return (x,y,z)

            elif varType == "boolean":

                response = self.sock.recv(1) # boolean = 1byte

                boolean = struct.unpack('b', response)

                return boolean

            # add a new vartype as distance (float) "dis"
            elif varType == "dis":
                response = self.sock.recv(4) # just one float， 4 bytes

                float = struct.unpack('f', response)

                return float

            elif varType == "xz":
                response = self.sock.recv(8)  # 2 (floats) * 4 (Bytes each) = 8 bytes

                # Convert the byte data back into x and z float values.
                x, z = struct.unpack('ff', response)

                return (x, z)

            
        except Exception as e:
            print(f"Error receiving state from Unity: {e}")
        
    def send_action_to_unity(self, key_code):
        """
        Sends a selected action to Unity via TCP.
        
        Parameters:
        - key_code (str): The key code representing the action to be sent to Unity.
        """
        
        try:
            # Convert the key code to bytes and sned the byte-formatted data to Unity.
            # keycodeByte = struct.pack('i', key_code)
            # # 1 (int) * 4 (Bytes) = 4 Bytes.
            key_code_str = str(key_code)
            keycodeByte = key_code_str.encode('utf-8')
            self.sock.sendall(keycodeByte)
            print("send:", keycodeByte)
        except Exception as e:
            print(f"Error sending action to Unity: {e}")
        
    
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    BELOW ARE METHODS FOR COMMUNICATION BETWEEN UNITY AND PYTHON SCRIPT
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        
    def connect_and_check_unity(self):
        """
        Initiates connection to Unity and ensures the connection is maintained.
        If the connection drops, the socket is closed.

        Returns:
        - bool: True if connected or connection is active, False otherwise.
        """
        try:
            if self.sock is None:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((self.host, self.port))
                print("Successfully connected to Unity!")
                return True
            else:
                return self.connection_check()
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return False

    def connection_check(self):
        """
        Validates the active connection to Unity.
        If the connection is inactive, the socket is closed.

        Returns:
        - bool: True if the connection is active, False otherwise.
        """
        try:
            self.sock.settimeout(0.1)
            self.sock.sendall(b'PING')
            response = self.sock.recv(12).decode("UTF-8")

            if response == 'DISCONNECTED':
                print("Disconnected with Unity")
                self.sock.close()
                self.sock = None
            elif response == 'PONG':
                print("Connection to Unity is still active.")
                return True
        except socket.error as e:
            print(f"Connection to Unity is lost. Reason: {e}")
            self.sock.close()
            self.sock = None


    def send_screenshot_message(self):
        self.sock.sendall(b'Screenshot')
        print("Take picture.")

    def send_done_message(self):
        """ send the end message of an episode"""
        self.sock.sendall(b'done')
        print("The episode ended.")

    def send_message_training_over(self):
        """
        Notifies Unity that training is completed.
        """
        self.sock.sendall(b'endOfTraining')
        print("Signaled Unity that training is complete.")
        print("Built File will be quited (in case of Unity editor, play mode will be quited).")
        

