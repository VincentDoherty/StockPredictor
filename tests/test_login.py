import unittest
from app import app, get_db_connection
from werkzeug.security import generate_password_hash

class FlaskTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

        # Set up the database and create a test user
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS users (id SERIAL PRIMARY KEY, username VARCHAR(255) UNIQUE NOT NULL, password_hash TEXT NOT NULL)")
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s) ON CONFLICT (username) DO NOTHING", ('testuser', generate_password_hash('testpassword')))
        conn.commit()
        cursor.close()
        conn.close()

    def tearDown(self):
        # Clean up the specific user
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE username = %s", ('testuser',))
        conn.commit()
        cursor.close()
        conn.close()

    def test_login(self):
        # Test login with correct credentials
        response = self.app.post('/login', data=dict(username='testuser', password='testpassword'), follow_redirects=True)
        self.assertIn(b'plot_actual_html', response.data)  # Check for specific content on the /stock page

        # Test login with incorrect credentials
        response = self.app.post('/login', data=dict(username='testuser', password='wrongpassword'), follow_redirects=True)
        self.assertIn(b'Invalid username or password', response.data)  # Check for the flash message

if __name__ == '__main__':
    unittest.main()