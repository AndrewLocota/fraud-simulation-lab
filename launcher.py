import subprocess
import sys
import os
import webbrowser
import time
import socket
from pathlib import Path

def find_free_port():
    """Find a free port for Streamlit"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def main():
    print("🚀 Starting Fraud Simulation Lab...")
    print("=" * 50)
    
    # Get the directory where this executable is located
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        app_dir = os.path.dirname(sys.executable)
    else:
        # Running as script
        app_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set the working directory
    os.chdir(app_dir)
    
    # Find app.py
    app_path = os.path.join(app_dir, 'app.py')
    if not os.path.exists(app_path):
        print("❌ Error: app.py not found!")
        print(f"Looking in: {app_dir}")
        input("Press Enter to exit...")
        return
    
    # Find a free port
    port = find_free_port()
    
    # Start Streamlit
    print(f"📡 Starting server on port {port}...")
    print("🌐 Opening browser...")
    print("\n" + "=" * 50)
    print("🎯 Fraud Simulation Lab is starting...")
    print("🔧 If browser doesn't open, go to:")
    print(f"   http://localhost:{port}")
    print("=" * 50)
    print("\n⚠️  IMPORTANT: Keep this window open while using the app!")
    print("❌ Close this window to stop the application.\n")
    
    try:
        # Start Streamlit in background
        process = subprocess.Popen([
            sys.executable, '-m', 'streamlit', 'run', 'app.py',
            '--server.port', str(port),
            '--server.headless', 'true',
            '--browser.gatherUsageStats', 'false'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Open browser
        webbrowser.open(f'http://localhost:{port}')
        
        # Keep the process running
        print("✅ Application is running!")
        print("📝 Processing logs:")
        print("-" * 30)
        
        # Wait for process to complete
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        process.terminate()
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main() 