var express = require('express')
var app = express();
let {PythonShell} = require('python-shell');
var util = require("util");
//const spawn = require("child_process").spawn;
//const pyProcess = spawn('python', ['comparesongs.py']);
var http = require('http');
var formidable = require('formidable');
var fs = require('fs');
//http.createServer
http.createServer(function (req, res) {
  if (req.url == '/fileupload') {
    var form = new formidable.IncomingForm();
    form.parse(req, function (err, fields, files) {
      var oldpath = files.filetoupload.path;
      var newpath = '/mnt/c/Users/Brandon/gitprojects/Honors-Thesis/Uploads/' + files.filetoupload.name;
      fs.copyFile(oldpath, newpath, function (err) {
        if (err) throw err;
        //res.write('File uploaded and moved! ');
	//res.write('Generating Song Recommendation... ');
	util.log('before python called');
	/*pyProcess.stdout.on('data', function(data) {
	  console.log(data.toString());
	  util.log(data.toString());
	  res.send(data.toString());
	  res.write(data.toString());
	});*/
	PythonShell.run('comparesongs.py', {
    	  mode: 'text',
    	  pythonOptions: ['-W ignore'],
    	  //args: [req.body.op, req.body.text],
    	  pythonPath: 'python3'
  	}, function(err, result) {
    	     if(err) {
      		console.log(err);
      		res.sendStatus(500);
    	     }
    	     else {
      		console.log('Success');
		console.log(result);
		//res.end();
      		res.send(result.join('\n'));
    	     }
  	});
	
	/*ps.run('comparesongs.py', null, function (err, results) {
  	  if (err) throw err;
  	  console.log('results: %j', results);
	  res.write('hey it worked');
	  res.write(results);
	});*/
	res.end();
      });
 });
  } else {
    res.writeHead(200, {'Content-Type': 'text/html'});
    res.write('<h1>Song Recommender</h1>');
    res.write('<form action="fileupload" method="post" enctype="multipart/form-data">');
    res.write('<input type="file" name="filetoupload"><br><br>');
    res.write('<input type="submit">');
    res.write('</form>');
    /*util.log('starting script');
    pyProcess.stdout.on('data', function(data) {
      console.log(data);
      util.log(data);
      util.log(data.toString());
      res.send(data.toString());
      res.write(data.toString());
    });
    util.log('finished script');*/
    return res.end();
  }
}).listen(8080);
