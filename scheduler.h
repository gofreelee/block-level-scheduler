#include <queue>
#include <vector>
#include <string>
class scheduler{

    public:
       static std::queue<std::string>  message_q; 
       
       void poll_from_queue();

    private:
};