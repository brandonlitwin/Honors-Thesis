var express = require('express');
var app = express();
let {PythonShell} = require('python-shell');
var util = require("util");
var http = require('http');
var formidable = require('formidable');
var fs = require('fs');
http.createServer(function (req, res) {
  if (req.url == '/fileupload') {
    var form = new formidable.IncomingForm();
    form.parse(req, function (err, fields, files) {
      var oldpath = files.filetoupload.path;
      var newpath = '/mnt/c/Users/Brandon/gitprojects/Honors-Thesis/Uploads/' + files.filetoupload.name;
      fs.copyFile(oldpath, newpath, function (err) {
        if (err) throw err;
	
	PythonShell.run('comparesongs.py', {
    	  mode: 'text',
    	  pythonOptions: ['-W ignore'],
    	  pythonPath: 'python3'
  	}, function(err, result) {
    	     if(err) {
      		console.log(err);
      		res.sendStatus(500);
    	     }
    	     else {
		console.log("Success");
                res.writeHead(200, {'Content-Type': 'text/html'});
	        var imgfile = '/mnt/c/Users/Brandon/gitprojects/Honors-Thesis/' + result
		
		fs.readFile(__dirname+'/'+result, function(err, data) { 
		  if (err) throw err;
                  res.writeHead(200, {'Content-Type': 'text/html'});
		  res.write('<html><body><img style="width:50%; height:100%;" src="data:image/png;base64,');
		  res.write(Buffer.from(data).toString('base64'));
		  res.end('"/></body></html>');
		});

    	     }
  	});
	

      });
 });
  } else {
    res.writeHead(200, {'Content-Type': 'text/html'});
    res.write('<h1>Song Recommender</h1>');
    res.write('<form action="fileupload" method="post" enctype="multipart/form-data">');
    res.write('<input type="file" name="filetoupload"><br><br>');
    res.write('<input type="submit">');
    res.write('</form>');
   
  }
}).listen(8080);
