document.addEventListener('DOMContentLoaded', function() {
    // Add click handlers for example queries
    const exampleQueries = document.querySelectorAll('.example-query');
    exampleQueries.forEach(query => {
        query.addEventListener('click', function() {
            document.getElementById('query').value = this.textContent.trim();
            // Scroll to search box
            document.getElementById('query').scrollIntoView({ behavior: 'smooth' });
            // Focus on the input
            setTimeout(() => document.getElementById('query').focus(), 500);
        });
    });
    
    // Format the results from Markdown to HTML when they're displayed
    const recommendationsElement = document.querySelector('.recommendations');
    if (recommendationsElement && recommendationsElement.textContent) {
        // Simple markdown-like formatting
        let content = recommendationsElement.innerHTML;
        
        // Convert markdown headers
        content = content.replace(/^# (.*$)/gim, '<h1>$1</h1>');
        content = content.replace(/^## (.*$)/gim, '<h2>$1</h2>');
        content = content.replace(/^### (.*$)/gim, '<h3>$1</h3>');
        
        // Convert bold and italic
        content = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        content = content.replace(/\*(.*?)\*/g, '<em>$1</em>');
        
        // Convert links
        content = content.replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" target="_blank">$1</a>');
        
        // Convert lists
        content = content.replace(/^- (.*$)/gim, '<li>$1</li>');
        
        // Wrap lists
        let inList = false;
        const lines = content.split('\n');
        for (let i = 0; i < lines.length; i++) {
            if (lines[i].startsWith('<li>') && !inList) {
                lines[i] = '<ul>' + lines[i];
                inList = true;
            } else if (!lines[i].startsWith('<li>') && inList) {
                lines[i-1] = lines[i-1] + '</ul>';
                inList = false;
            }
        }
        if (inList) {
            lines.push('</ul>');
        }
        
        recommendationsElement.innerHTML = lines.join('\n');
    }
});