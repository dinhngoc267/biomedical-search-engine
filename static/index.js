window.onload = function() {
    document.getElementById("wiki-search-input").focus();

	$("#btn-search").click(function(event) {
		event.preventDefault();
		alert('hi')
		var keyword = $(".wiki-search-input").val();
		if (keyword !== "") {
	
			$(".result-wiki-search-form-input").val(keyword);
			$(".home").addClass('hidden');
				$(".result").removeClass('hidden');
			document.getElementById("wiki-search-input").blur();
			   $(".wiki-search-input").val("");
			document.getElementById("result-wiki-search-form-input").blur();	
			$(".display-results").html("");
			ajax(keyword);
		}
	
		else {
			alert("Enter a keyword into the search box");
		}
		
	});
};

// function ajax (keyword) { //AJAX request
// 	$.ajax({ 
// 		url: "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=" + keyword + "&prop=info&inprop=url&utf8=&format=json",
// 		dataType: "jsonp",
// 		success: function(response) {
// 			console.log(response.query);
// 			if (response.query.searchinfo.totalhits === 0) {
// 				showError(keyword);
// 			}

// 			else {
// 				showResults(response);
// 			}
// 		},
// 		error: function () {
// 			alert("Error retrieving search results, please refresh the page");
// 		}
// 	});
// }

function ajax (keyword) { //AJAX request
	$.ajax({ 
		cache: false,
		url: "/",
		type: 'POST',
		contentType: "application/json",
		dataType: "json",
		data: JSON.stringify({'query':keyword}),
		success: function(response) {
			if (response.result === 0) {
				showError(keyword);
			}

			else {
				showResults(response.data);
			}
		},
		error: function () {
			alert("Error retrieving search results, please refresh the page");
		}
	});
}

function showResults (callback) {
	console.log(callback)
	// for (var i = 0; i <= 9; i++) {
	// 	$(".display-results").append("<div class='result-list result-" + i + "'>" + "<span class='result-title title-" + i + "'></span>" + "<br>" +"<span class='result-snippet snippet-" + i + "'></span>" + "<br>" + "<span class='result-metadata metadata-" + i + "'></span>" + "</div>" );
	// }

	// for (var m = 0; m <= 9; m++) {
	// 	var title = callback.query.search[m].title;
	// 	var url = title.replace(/ /g, "_");
	// 	var timestamp = callback.query.search[m].timestamp;
	// 	timestamp = new Date(timestamp);
	// 	//"Wed Aug 27 2014 00:27:15 GMT+0100 (WAT)";
	// 	console.log(timestamp);
	// 	$(".title-" + m).html("<a href='https://en.wikipedia.org/wiki/" + url + "' target='_blank'>" + callback.query.search[m].title + "</a>");
	// 	$(".snippet-" + m).html(callback.query.search[m].snippet);
	// 	$(".metadata-" + m).html((callback.query.search[m].size/1000).toFixed(0) + "kb (" + callback.query.search[m].wordcount + " words) - " + timestamp);
	// }
}

function showError(keyword) {
	$(".display-results").append( "<div class='error'> <p>Your search <span class='keyword'>" + keyword + "</span> did not match any documents.</p> <p>Suggestions:</p><li>Make sure that all words are spelled correctly.</li><li>Try different keywords.</li><li>Try more general keywords.</li></div> ");
}


$(".result-btn-wiki").click(function (event) {
	alert('hi')
	event.preventDefault();
	$(".display-results").html("");
	var keyword = $(".result-wiki-search-form-input").val();
	document.getElementById("result-wiki-search-form-input").blur();
	ajax(keyword);
});
