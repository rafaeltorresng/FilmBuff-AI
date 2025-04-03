from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import uuid
import threading
import time
from datetime import datetime
from Movies_Agent import get_recommendations

app = Flask(__name__)

# Store ongoing and completed tasks
tasks = {}

@app.route('/')
def index():
    """Render the home page with search form"""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Process search requests and start recommendation task"""
    query = request.form.get('query', '')
    include_people = request.form.get('include_people') == 'on'
    
    if not query:
        return render_template('index.html', error="Please enter a search query")
    
    # Generate a unique task ID
    task_id = str(uuid.uuid4())
    
    # Store task info
    tasks[task_id] = {
        'query': query,
        'include_people': include_people,
        'status': 'processing',
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'result': None
    }
    
    # Start recommendation process in background thread
    threading.Thread(
        target=process_recommendations,
        args=(task_id, query, include_people)
    ).start()
    
    # Redirect to results page
    return redirect(url_for('show_results', task_id=task_id))

def process_recommendations(task_id, query, include_people):
    """Background process to get recommendations"""
    try:
        result = get_recommendations(query, include_people)
        tasks[task_id]['status'] = 'completed'
        tasks[task_id]['result'] = result
    except Exception as e:
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['error'] = str(e)

@app.route('/results/<task_id>')
def show_results(task_id):
    """Display recommendation results"""
    if task_id not in tasks:
        return render_template('index.html', error="Invalid task ID")
    
    task = tasks[task_id]
    return render_template('results.html', task=task, task_id=task_id)

@app.route('/api/status/<task_id>')
def task_status(task_id):
    """API endpoint for checking task status"""
    if task_id not in tasks:
        return jsonify({'status': 'not_found'})
    
    return jsonify({
        'status': tasks[task_id]['status'],
        'query': tasks[task_id]['query']
    })

# Clean up old tasks periodically (in a production app, use a proper job scheduler)
@app.before_request
def cleanup_old_tasks():
    current_time = time.time()
    # Run cleanup approximately once per hour
    if int(current_time) % 3600 < 10:
        for task_id in list(tasks.keys()):
            task_time = datetime.strptime(tasks[task_id]['created_at'], '%Y-%m-%d %H:%M:%S')
            # Remove tasks older than 24 hours
            if (datetime.now() - task_time).total_seconds() > 86400:
                tasks.pop(task_id, None)

if __name__ == '__main__':
    app.run(debug=True)
