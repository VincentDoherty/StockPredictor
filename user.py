from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin

class User(UserMixin):
    def __init__(self, id, username, password_hash, role='user'):
        self.id = id
        self.username = username
        self.password_hash = password_hash
        self.role = role

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def has_permission(self, permission):
        # Define permissions based on roles
        role_permissions = {
            'admin': ['view', 'edit', 'delete'],
            'user': ['view', 'edit']
        }
        return permission in role_permissions.get(self.role, [])