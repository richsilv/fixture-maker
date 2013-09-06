$(function() {
    
    $('#submit').on("click", function() {
        $.post('/get_fixtures/', {'teams': $('#teams').val(), 'groups': $('#groups').val(), 'pitches': $('#pitches').val(), 'reverse': $('#reverse').is(':checked'),
                                  'optrounds': $('#optrounds').is(':checked'), 'optpitches': $('#optpitches').is(':checked')}, function(r) {  
            $('#results').html(r);    
            $('#download').css("display", "block");  
            });            
        });    

});