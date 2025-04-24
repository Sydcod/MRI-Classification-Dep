#!/usr/bin/env python
import os
import sys
import subprocess
import time
import signal
import atexit
import webbrowser
from urllib.request import urlopen
from urllib.error import URLError

# Process holders
frontend_process = None
backend_process = None

def is_port_in_use(port):
    """Check if a port is in use"""
    try:
        urlopen(f"http://localhost:{port}", timeout=1)
        return True
    except URLError:
        return False

def clean_up():
    """Terminate all processes when the script exits"""
    print("\nShutting down all services...")
    
    if backend_process:
        print("Stopping backend server...")
        if os.name == 'nt':  # Windows
            subprocess.call(['taskkill', '/F', '/T', '/PID', str(backend_process.pid)])
        else:  # Unix/Linux/MacOS
            os.killpg(os.getpgid(backend_process.pid), signal.SIGTERM)
    
    if frontend_process:
        print("Stopping frontend server...")
        if os.name == 'nt':  # Windows
            subprocess.call(['taskkill', '/F', '/T', '/PID', str(frontend_process.pid)])
        else:  # Unix/Linux/MacOS
            os.killpg(os.getpgid(frontend_process.pid), signal.SIGTERM)
            
    print("All services have been stopped.")

def start_backend():
    """Start the Flask backend server"""
    global backend_process
    
    # Check if port is already in use
    if is_port_in_use(5000):
        print("Port 5000 is already in use. Backend may already be running.")
        return False
    
    print("Starting backend server...")
    os.chdir('backend')
    
    # Start the Flask app
    if os.name == 'nt':  # Windows
        backend_process = subprocess.Popen(
            ['python', 'app.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:  # Unix/Linux/MacOS
        backend_process = subprocess.Popen(
            ['python', 'app.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid
        )
    
    os.chdir('..')
    
    # Wait for the backend to start
    print("Waiting for backend to start...")
    start_time = time.time()
    while not is_port_in_use(5000):
        time.sleep(1)
        if time.time() - start_time > 30:  # Timeout after 30 seconds
            print("Backend server failed to start within 30 seconds.")
            return False
        
    print("Backend server is running on http://localhost:5000")
    return True

def start_frontend():
    """Start the React frontend development server"""
    global frontend_process
    
    # Check if port is already in use
    if is_port_in_use(3000):
        print("Port 3000 is already in use. Frontend may already be running.")
        return False
    
    print("Starting frontend server...")
    os.chdir('frontend')
    
    # Start the React development server
    if os.name == 'nt':  # Windows
        frontend_process = subprocess.Popen(
            ['npm', 'start'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:  # Unix/Linux/MacOS
        frontend_process = subprocess.Popen(
            ['npm', 'start'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid
        )
    
    os.chdir('..')
    
    # Wait for the frontend to start
    print("Waiting for frontend to start...")
    start_time = time.time()
    while not is_port_in_use(3000):
        time.sleep(1)
        if time.time() - start_time > 60:  # Timeout after 60 seconds
            print("Frontend server failed to start within 60 seconds.")
            return False
            
    print("Frontend server is running on http://localhost:3000")
    return True

def main():
    """Main function to run the application"""
    print("Starting Brain MRI Classification Web Application...")
    
    # Register cleanup handler
    atexit.register(clean_up)
    
    # Start the backend first
    if start_backend():
        # Then start the frontend
        if start_frontend():
            print("\nBoth servers are running!")
            print("Frontend: http://localhost:3000")
            print("Backend API: http://localhost:5000")
            
            # Open the web app in the default browser
            time.sleep(2)  # Give the servers a bit more time to stabilize
            webbrowser.open('http://localhost:3000')
            
            # Keep the script running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nReceived keyboard interrupt. Shutting down...")
        else:
            print("Failed to start frontend server.")
    else:
        print("Failed to start backend server.")

if __name__ == "__main__":
    main() 