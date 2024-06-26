function getLevel(result) {
    switch (result) {
        case "Baik":
            return { 
                name: "Good", 
                class: "level-baik", 
                desc:  "The air quality level is very good, has no negative effects on humans, animals and plants.", 
                recom: "-" 
            };
            break;
        case "Sedang":
            return { 
                name: "Medium", 
                class: "level-sedang", 
                desc: "Air quality levels are still acceptable for human, animal and plant health.", 
                recom: "People with vulnerable physical conditions should reduce outdoor activities and/or work hard." 
            };
            break;
        case "Tidak Sehat":
            return { 
                name: "Unhealthy", 
                class: "level-tidak-sehat", 
                desc: "Air quality levels that are detrimental to humans, animals and plants.", 
                recom: "-" 
            };
            break;
        case "SANGAT TIDAK SEHAT":
            return { 
                name: "Very Unhealthy", 
                class: "level-sangat-tidak-sehat", 
                desc: "Air quality levels that can increase health risks in a number of exposed segments of the population.", 
                recom: "-" 
            };
            break;
        case "BERBAHAYA":
            return { name: "Dangerous!", 
            class: "level-berbahaya", 
            desc: "Air quality levels that can seriously harm the health of the population and require immediate treatment.", 
            recom: "-" 
        };
            break;
        default:
            return { 
                name: "", 
                class: "", 
                desc: "", 
                recom: "" 
            };
    }
}

function updatePrediction(status) {
    var level = getLevel(status);

    document.getElementById("level-text").innerText = level.name;
    document.getElementById("level-description").innerText = level.desc;
    document.getElementById("level-recommendation").innerText = level.recom;

    
    var resultCardHeader = document.getElementById("result-card-header");
    resultCardHeader.className = "card-header p-3 " + level.class;
}

document.addEventListener("DOMContentLoaded", function() {
    var status = document.getElementById("output").innerText.trim();

    updatePrediction(status);
});