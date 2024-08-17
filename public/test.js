document.addEventListener('DOMContentLoaded', function() {
    function overrideFooterElement() {
        const footerElement = document.querySelector("#root > div > div > div > div.MuiStack-root.css-1ylu0bo > div.MuiStack-root.css-zpi9s5 > div > div > div.MuiBox-root.css-mww0i9 > div.MuiStack-root.watermark.css-1705j0v > a");
        if(footerElement) {
            const pElement = footerElement.querySelector('p');
            if(pElement) {
                pElement.textContent = 'example text';
            }
        }
    }

    setTimeout(overrideFooterElement, appropriate_time);
});